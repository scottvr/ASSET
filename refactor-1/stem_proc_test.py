project_root/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio.py           # AudioSegment and basic audio handling
│   │   ├── types.py           # Common types, dataclasses, enums
│   │   └── utils.py           # Common utilities
│   │
│   ├── separation/
│   │   ├── __init__.py
│   │   ├── base.py            # Base separator class
│   │   ├── spleeter.py        # Spleeter implementation
│   │   └── factory.py         # Separator factory/registry
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── base.py            # Base analyzer class
│   │   ├── spectral.py        # Spectral analysis
│   │   ├── phase.py           # Phase analysis
│   │   └── visualization.py    # Plotting utilities
│   │
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── models.py          # LoRA and phase-aware models
│   │   ├── training.py        # Training loops and logic
│   │   └── inference.py       # Inference pipeline
│   │
│   └── io/
│       ├── __init__.py
│       ├── audio.py           # Audio file handling
│       └── spectrograms.py    # Spectrogram generation/saving
│
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_separation.py
│   ├── test_analysis.py
│   └── test_diffusion.py
│
├── notebooks/
│   ├── separation_test.ipynb
│   └── diffusion_test.ipynb
│
└── examples/
    ├── basic_separation.py
    └── phase_aware_processing.py
