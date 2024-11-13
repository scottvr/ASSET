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
class ChimeraCat:
    def __init__(self, 
                 src_dir: str = "src", 
                 compression_level: CompressionLevel = CompressionLevel.FULL,
                 rules: Optional[CompressionRules] = None):
        self.src_dir = Path(src_dir)
        self.compression_level = compression_level
        self.rules = rules or CompressionRules.default_rules()
        self.modules: Dict[Path, ModuleInfo] = {}
        self.dep_graph = nx.DiGraph()
        
    def analyze_file(self, file_path: Path) -> ModuleInfo:
        """Analyze a Python file for imports and definitions"""
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

    def _get_sorted_files(self) -> List[Path]:
        """Get files sorted by dependencies"""
        try:
            return list(nx.topological_sort(self.dep_graph))
        except nx.NetworkXUnfeasible:
            print("Warning: Circular dependencies detected. Using simple ordering.")
            return list(self.modules.keys()) 
    
    def build_dependency_graph(self):
        """Build a dependency graph of all Python files"""
        # Find all Python files
        for file_path in self.src_dir.rglob("*.py"):
            if file_path.name != "__init__.py":
                module_info = self.analyze_file(file_path)
                self.modules[file_path] = module_info
                self.dep_graph.add_node(file_path)
        
        # Add edges for dependencies
        for file_path, module in self.modules.items():
            pkg_path = file_path.relative_to(self.src_dir).parent
            for imp in module.imports:
                # Convert import to potential file paths
                imp_parts = imp.split('.')
                for other_path in self.modules:
                    other_pkg = other_path.relative_to(self.src_dir).parent
                    if str(other_pkg) == '.'.join(imp_parts[:-1]):
                        self.dep_graph.add_edge(file_path, other_path)
    
    def _compress_content(self, content: str) -> str:
        """Apply compression based on current level"""
        if self.compression_level == CompressionLevel.FULL:
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

        # Generate combined file with compression
        output = [
            header,
            "# External imports",
            *self._get_external_imports(),
            "\n# Combined module code\n"
        ]
        
        for file_path in self._get_sorted_files():
            module = self.modules[file_path]
            rel_path = file_path.relative_to(self.src_dir)
            
            output.extend([
                f"\n# From {rel_path}",
                self._compress_content(module.content)
            ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))
            
        return output_file
        
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
                        "```\n",
                        " /\\___/\\  ChimeraCat\n",
                        "( o   o )  Modular Python Fusion\n",
                        "(  =^=  )\n",
                        " (_____)  Generated: {timestamp}\n",
                        "```\n"
                    ],
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": code.split('\n'),
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Usage Example"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Example usage",
                        "cat = ChimeraCat('srcdir')",
                        "notebook_file = cat.generate_colab_notebook()",
                        "print(f\"Generated notebook: {notebook_file}\")",
                        "cat = ChimeraCat('srcdir', compression_level = CompressionLevel.ESSENTIAL: ",
                        "output_file = cat.generate_concat_file(\"essential_code.py\")",
                 
                    ],
                    "execution_count": None,
                    "outputs": []
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
    # Example with different compression levels
    examples = {
        CompressionLevel.SIGNATURES: "signatures_only.py",
        CompressionLevel.ESSENTIAL: "essential_code.py",
        CompressionLevel.VERBOSE: "verbose_code.py",
        CompressionLevel.FULL: "full_code.py"
    }
    
    for level, filename in examples.items():
        cat = ChimeraCat("src", compression_level=level)
        output_file = cat.generate_concat_file(filename)
        print(f"Generated {level.value} version: {output_file}")
    
    cat = ChimeraCat("src")
    output_file = cat.generate_colab_notebook("colab_combined.ipynb")
    print(f"Generated notebook:{output_file}")

if __name__ == "__main__":
    main()
