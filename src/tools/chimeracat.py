"""
ChimeraCat arose as an ancillary utility for some larger work I was 
doing with the help of Claude 3.5 Sonnet (New) in October 2024 

This utility:

Analyzes Python files for imports and definitions
Builds a dependency graph
Generates both a single .py file and a Colab notebook
Handles internal vs external imports
Avoids duplicate definitions
Creates a clean, organized output
Adds usage examples

```python
from ChimeraCat import ChimeraCat

# Generate both notebook and Python file
concat = ChimeraCat("src")
notebook_file = concat.generate_colab_notebook()
py_file = concat.generate_concat_file()
```

Features Claude is particularly proud of:
- Dependency ordering using NetworkX
- Duplicate prevention
- Clean handling of internal vs external imports
- Automatic notebook generation
- Maintains code readability with section headers
"""

import re
from pathlib import Path
from typing import List, Set, Dict
import networkx as nx
from dataclasses import dataclass
from datetime import datetime

from enum import Enum
from typing import Dict, List, Set, Optional, Pattern
import re
from dataclasses import dataclass, field
from datetime import datetime

class CompressionLevel(Enum):
    SIGNATURES = "signatures"  # Just interfaces/types/docstrings
    ESSENTIAL = "essential"    # + Core logic, skip standard patterns
    VERBOSE = "verbose"        # Everything except obvious boilerplate
    FULL = "full"             # Complete code

@dataclass
class CompressionPattern:
    """Pattern for code compression with explanation"""
    pattern: str
    replacement: str
    explanation: str
    flags: re.RegexFlag = re.MULTILINE

    def apply(self, content: str) -> str:
        return re.sub(self.pattern, f"{self.replacement} # {self.explanation}\n", 
                     content, flags=self.flags)

@dataclass
class CompressionRules:
    """Collection of compression patterns for different levels"""
    signatures: List[CompressionPattern] = field(default_factory=list)
    essential: List[CompressionPattern] = field(default_factory=list)
    verbose: List[CompressionPattern] = field(default_factory=list)

    @classmethod
    def default_rules(cls) -> 'CompressionRules':
        return cls(
            signatures=[
                CompressionPattern(
                    pattern=r'(class|def)\s+\w+[^:]*:\s*(?:"""(?:.*?)""")?.*?(?=(?:class|def|\Z))',
                    replacement=r'\1\n    ...',
                    explanation="Implementation details elided",
                    flags=re.MULTILINE | re.DOTALL
                )
            ],
            essential=[
                CompressionPattern(
                    pattern=r'def get_\w+\(.*?\):\s*return [^;{}]+?\n',
                    replacement='    ...',
                    explanation="Simple getter method"
                ),
                CompressionPattern(
                    pattern=r'def __init__\(self(?:,\s*[^)]+)?\):\s*(?:[^{};]+?\n\s+)+?(?=\n\s*\w|$)',
                    replacement='    ...',
                    explanation="Standard initialization"
                )
            ],
            verbose=[
                CompressionPattern(
                    pattern=r'if __name__ == "__main__":\s*(?:[^{};]+?\n\s*)+?(?=\n\s*\w|$)',
                    replacement='    ...',
                    explanation="Main execution block"
                ),
                CompressionPattern(
                    pattern=r'def __str__\(self\):\s*return [^;{}]+?\n',
                    replacement='    ...',
                    explanation="String representation"
                )
            ]
        )

@dataclass
class ModuleInfo:
    """Information about a Python module"""
    path: Path
    content: str
    imports: Set[str]
    classes: Set[str]
    functions: Set[str]

class ChimeraCat:
    """Utility to concatenate modular code into Colab-friendly single files"""
    def __init__(self, 
                 src_dir: str = "src", 
                 compression_level: CompressionLevel = CompressionLevel.FULL,
                 exclude_patterns: List[str] = None, rules: Optional[CompressionRules] = None,
                 debug: bool = False):
        self.src_dir = Path(src_dir)
        self.compression_level = compression_level
        self.rules = rules or CompressionRules.default_rules()
        self.modules: Dict[Path, ModuleInfo] = {}
        self.dep_graph = nx.DiGraph()
        self.self_path = Path(__file__).resolve()
        self.exclude_patterns = exclude_patterns or []
        self.debug = debug

    def _debug_print(self, *args, **kwargs):
        """Helper for debug output"""
        if self.debug:
            print(*args, **kwargs)

    def should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded from processing"""
        # Always exclude self
        if file_path.resolve() == self.self_path:
            return True
            
        # Check against exclude patterns
        str_path = str(file_path)
        return any(pattern in str_path for pattern in self.exclude_patterns)

    def analyze_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyze a Python file for imports and definitions"""
        if self.should_exclude(file_path):
            return None

        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find imports
        import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'
        imports = set()
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            if match.group(1):  # from X import Y
                imports.add(match.group(1))
            else:  # import X
                imports.add(match.group(2).split(',')[0].strip())
                
        # Find class definitions
        class_pattern = r'class\s+(\w+)'
        classes = set(re.findall(class_pattern, content))
        
        # Find function definitions
        func_pattern = r'def\s+(\w+)'
        functions = set(re.findall(func_pattern, content))
        
        return ModuleInfo(
            path=file_path,
            content=content,
            imports=imports,
            classes=classes,
            functions=functions
        )
    
    def _process_imports(self, content: str, module_path: Path) -> str:
        """Process and adjust imports for concatenated context"""
        lines = []
        for line in content.splitlines():
            if line.strip().startswith('from .') or line.strip().startswith('from ..'):
                # Convert relative import to docstring and preserve indentation
                indent = len(line) - len(line.lstrip())
                spaces = ' ' * indent
                lines.append(f'{spaces}"""{line.strip()}  # Original relative import"""')
            else:
                lines.append(line)
        return '\n'.join(lines)
    
    def build_dependency_graph(self):
        """Build a dependency graph of all Python files"""
        # First pass: Collect all modules and create nodes
        for file_path in self.src_dir.rglob("*.py"):
            module_info = self.analyze_file(file_path)
            if module_info is not None:
                self.modules[file_path] = module_info
                # Each file is a node in our graph
                self.dep_graph.add_node(file_path)
        
        # Second pass: Add edges based on imports
        for file_path, module in self.modules.items():
            pkg_path = file_path.relative_to(self.src_dir).parent
            for imp in module.imports:
                # Convert import to potential file paths
                imp_parts = imp.split('.')
                
                # Handle different import types
                if imp.startswith('.'):  # Relative import
                    # Calculate actual path from relative import
                    relative_depth = imp.count('.')
                    target_path = pkg_path
                    for _ in range(relative_depth - 1):
                        target_path = target_path.parent
                    remaining_parts = imp_parts[relative_depth:]
                else:  # Absolute import
                    remaining_parts = imp_parts

                # Find matching module for this import
                for other_path in self.modules:
                    other_pkg = other_path.relative_to(self.src_dir).parent
                    if self._paths_match(other_pkg, remaining_parts):
                        # Add directed edge: file_path depends on other_path
                        self.dep_graph.add_edge(file_path, other_path)

                        print(f"    Added dependency: {file_path.name} -> {other_path.name}")

        
    def generate_concat_file(self, output_file: str = "colab_combined.py") -> str:
        """Generate a single file combining all modules in dependency order"""
        self.build_dependency_graph()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    
        header = f"""# Generated by ChimeraCat
    #  /\\___/\\  ChimeraCat
    # ( o   o )  Modular Python Fusion
    # (  =^=  ) 
    #  (______)  Generated: {timestamp}
    #
    # Compression Level: {self.compression_level.value}
    """
        
        # Start with external imports
        output = [
            header,
            "# External imports",
            *self._get_external_imports(),
            "\n# Combined module code\n"
        ]
        
        # Get files in dependency order
        sorted_files = self._get_sorted_files()
        
        # Create a map of original module paths to their contents
        module_contents = {}
        
        # First pass: collect and process all module contents
        for file_path in sorted_files:
            if file_path in self.modules:
                module = self.modules[file_path]
                rel_path = file_path.relative_to(self.src_dir)
                
                # Process imports and compress content
                processed_content = self._process_imports(
                    self._compress_content(module.content),
                    file_path
                )
                
                module_contents[file_path] = {
                    'content': processed_content,
                    'rel_path': rel_path
                }
        
        # Second pass: output in correct order with headers
        for file_path in sorted_files:
            if file_path in module_contents:
                info = module_contents[file_path]
                output.extend([
                    f"\n# From {info['rel_path']}",
                    info['content']
                ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))
            
        return output_file
    
    def _get_external_imports(self) -> List[str]:
      """Get sorted list of external imports from all modules"""
      external_imports = set()
      for module in self.modules.values():
          external_imports.update(
              imp for imp in module.imports 
              if not any(str(imp).startswith(str(p.relative_to(self.src_dir).parent)) 
                        for p in self.modules)
              and not imp.startswith('.')
          )
      
      # Format and sort the import statements
      return sorted(f"import {imp}" for imp in external_imports)

    def _paths_match(self, path: Path, import_parts: List[str]) -> bool:
        """Check if a path matches an import statement"""
        path_parts = list(path.parts)
        return len(path_parts) == len(import_parts) and \
               all(p == i for p, i in zip(path_parts, import_parts))

    def _get_sorted_files(self) -> List[Path]:
        """Get files sorted by dependencies"""
        try:
            # Topological sort ensures dependencies come before dependents
            sorted_files = list(nx.topological_sort(self.dep_graph))
            
            # Debug info
            print("\nDependency Resolution:")
            for idx, file in enumerate(sorted_files):
                deps = list(self.dep_graph.predecessors(file))
                print(f"{idx+1}. {file.name}")
                if deps:
                    print(f"   Depends on: {', '.join(d.name for d in deps)}")
            
            return sorted_files

        except nx.NetworkXUnfeasible as e:
            # If we detect a cycle, identify and report it
            cycles = list(nx.simple_cycles(self.dep_graph))
            print("Warning: Circular dependencies detected:")
            for cycle in cycles:
                cycle_path = ' -> '.join(p.name for p in cycle)
                print(f"  {cycle_path}")
            
            # Fall back to simple ordering but warn user
            print("Using simple ordering instead.")
            return list(self.modules.keys())

    def visualize_dependencies(self, output_file: str = "dependencies.png"):
        """Optional: Visualize the dependency graph"""
        try:
            import matplotlib.pyplot as plt
            pos = nx.spring_layout(self.dep_graph)
            plt.figure(figsize=(12, 8))
            nx.draw(self.dep_graph, pos, with_labels=True, 
                   labels={p: p.name for p in self.dep_graph.nodes()},
                   node_color='lightblue',
                   node_size=2000,
                   font_size=8)
            plt.savefig(output_file)
            plt.close()
            return output_file
        except ImportError:
            print("matplotlib not available for visualization")
            return None

#     def _get_sorted_files(self) -> List[Path]:
#         """Get files sorted by dependencies with optional logging"""
#         try:
#             sorted_files = list(nx.topological_sort(self.dep_graph))
#             
#             if self.debug:
#                 self._debug_print("\nResolved module order:")
#                 for idx, file in enumerate(sorted_files):
#                     deps = list(self.dep_graph.predecessors(file))
#                     rel_path = file.relative_to(self.src_dir)
#                     self._debug_print(f"{idx+1}. {rel_path}")
#                     if deps:
#                         dep_paths = [d.relative_to(self.src_dir) for d in deps]
#                         self._debug_print(f"   Depends on: {', '.join(map(str, dep_paths))}")
#             
#             return sorted_files
#             
#         except nx.NetworkXUnfeasible:
#             self._debug_print("\nWarning: Circular dependencies detected!")
#             # Always print circular dependency warnings regardless of debug mode
#             print("Warning: Circular dependencies found in modules. Check dependency report for details.")
#             if self.debug:
#                 for cycle in nx.simple_cycles(self.dep_graph):
#                     cycle_paths = [p.relative_to(self.src_dir) for p in cycle]
#                     self._debug_print(f"  {' -> '.join(map(str, cycle_paths))}")
#             
#             return list(self.modules.keys())
#     
#     def visualize_dependencies(self, output_file: str = "dependencies.png"):
#         """Visualize the dependency graph with detailed node information"""
#         try:
#             import matplotlib.pyplot as plt
#             
#             # Create figure
#             plt.figure(figsize=(12, 8))
#             
#             # Generate layout
#             pos = nx.spring_layout(self.dep_graph, k=2, iterations=50)
#             
#             # Draw nodes
#             nx.draw_networkx_nodes(self.dep_graph, pos,
#                                  node_color='lightblue',
#                                  node_size=2000)
#             
#             # Draw edges with arrows
#             nx.draw_networkx_edges(self.dep_graph, pos,
#                                  edge_color='gray',
#                                  arrows=True,
#                                  arrowsize=20)
#             
#             # Add labels with relative paths
#             labels = {p: str(p.relative_to(self.src_dir)) for p in self.dep_graph.nodes()}
#             nx.draw_networkx_labels(self.dep_graph, pos, labels,
#                                   font_size=8,
#                                   font_weight='bold')
#             
#             # Add title and information
#             plt.title("Module Dependencies\n", pad=20)
#             
#             # Add dependency statistics
#             plt.figtext(0.02, 0.02, 
#                        f"Total Modules: {len(self.modules)}\n"
#                        f"Dependencies: {self.dep_graph.number_of_edges()}\n"
#                        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
#                        fontsize=8)
#             
#             # Save with high DPI for clarity
#             plt.savefig(output_file, dpi=300, bbox_inches='tight')
#             plt.close()
#             
#             self._debug_print(f"\nDependency graph visualization saved to: {output_file}")
#             return output_file
#             
#         except ImportError:
#             print("matplotlib not available for visualization")
#             return None
# 
    def get_dependency_report(self) -> str:
        """Generate a detailed dependency report"""
        report = ["Dependency Analysis Report", "=" * 25, ""]
        
        # Module statistics
        report.extend([
            "Module Statistics:",
            f"Total modules: {len(self.modules)}",
            f"Total dependencies: {self.dep_graph.number_of_edges()}",
            ""
        ])
        
        # Dependency chains
        report.extend(["Dependency Chains:", "-" * 17])
        try:
            sorted_files = list(nx.topological_sort(self.dep_graph))
            for idx, file in enumerate(sorted_files):
                deps = list(self.dep_graph.predecessors(file))
                report.append(f"{idx+1}. {file.relative_to(self.src_dir)}")
                if deps:
                    report.append(f"   Depends on: {', '.join(str(d.relative_to(self.src_dir)) for d in deps)}")
            report.append("")
        except nx.NetworkXUnfeasible:
            report.extend([
                "Warning: Circular dependencies detected!",
                "Cycles found:",
                *[f"  {' -> '.join(str(p.relative_to(self.src_dir)) for p in cycle)}"
                  for cycle in nx.simple_cycles(self.dep_graph)],
                ""
            ])
        
        # Module details
        report.extend(["Module Details:", "-" * 13])
        for path, module in self.modules.items():
            report.extend([
                f"\n{path.relative_to(self.src_dir)}:",
                f"  Classes: {', '.join(module.classes) if module.classes else 'None'}",
                f"  Functions: {', '.join(module.functions) if module.functions else 'None'}",
                f"  Imports: {', '.join(module.imports) if module.imports else 'None'}"
            ])
        
        return '\n'.join(report)
    
    
    def _compress_content(self, content: str) -> str:
        """Apply compression based on current level"""
        if self.compression_level == CompressionLevel['FULL']:
            return content

        compressed = content
        patterns = []
        
        # Apply patterns based on level
        if self.compression_level == CompressionLevel.SIGNATURES:
            patterns = self.rules.signatures
        elif self.compression_level == CompressionLevel.ESSENTIAL:
            patterns = self.rules.signatures + self.rules.essential
        elif self.compression_level == CompressionLevel.VERBOSE:
            patterns = self.rules.verbose

        # Apply each pattern
        for pattern in patterns:
            compressed = pattern.apply(compressed)

        return compressed


    def generate_colab_notebook(self, output_file: str = "colab_combined.ipynb"):
        """Generate a Jupyter notebook with the combined code"""
        py_file = self.generate_concat_file("temp_combined.py")
        
        with open(py_file, 'r') as f:
            code = f.read()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "##Notebook Generated by ChimeraCat\n"
                    ],
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": code.splitlines(keepends=True),
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "```\n",
                        " /\\___/\\   ChimeraCat\n",
                        "( o   o )  Modular Python Fusion\n",
                        "(  =^=  )  https://github.com/scottvr/chimeracat\n",
                        f" (______)  Generated: {timestamp}\n",
                        "```\n"
                    ]
                }

            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        Path("temp_combined.py").unlink()  # Clean up temporary file
        return output_file

def main():
    debug = True
    # Example with different compression levels
    examples = {
        CompressionLevel.SIGNATURES: "signatures_only.py",
        CompressionLevel.ESSENTIAL: "essential_code.py",
        CompressionLevel.VERBOSE: "verbose_code.py",
        CompressionLevel.FULL: "full_code.py"
    }
    
    for level, filename in examples.items():
        cat = ChimeraCat("src", compression_level=level, debug=debug)
        output_file = cat.generate_concat_file(filename)
        print(f"Generated {level.value} version: {output_file}")
    
    cat = ChimeraCat("src", CompressionLevel.FULL, debug=debug)
    output_file = cat.generate_colab_notebook()
    print(f"Generated colab notebook  version: {output_file}")

    if debug:
        cat.visualize_dependencies("module_deps.png")
        # Get detailed report
        report = cat.get_dependency_report()
        print(report)
        # Generate visualization


if __name__ == "__main__":
    main()
