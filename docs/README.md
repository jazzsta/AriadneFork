# Ariadne Documentation

This directory contains detailed documentation for the Ariadne root system analysis software.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Scientific Methods](scientific-methods.md) | Pareto optimality calculations, formulas, and academic references |
| [Output Fields Reference](output-fields.md) | Complete reference for all CSV output fields with units and interpretation |
| [Ariadne 0.3 Research Notebook](../Ariadne0.3_Exploration.ipynb) | Reproducible figures and simulations from *Road to Ariadne 0.3* |

## Quick Links

- **Main README**: [../README.md](../README.md) - Installation and usage instructions
- **GitHub Repository**: https://github.com/Salk-Harnessing-Plants-Initiative/Ariadne
- **PyPI Package**: https://pypi.org/project/ariadne-roots/

## For Scientists

If you're using Ariadne for research and need to cite the underlying methods:

1. See [Scientific Methods](scientific-methods.md) for the mathematical foundations and primary references
2. The key papers are:
   - Chandrasekhar & Navlakha (2019) for Pareto optimality in biological networks
   - Conn et al. (2017) for network design principles in plant architectures

## For Developers

- Source code: `src/ariadne_roots/`
- Key modules:
  - `main.py` - Trace and analyze GUI
  - `quantify.py` - Root trait calculations, analysis pipeline, CSV/scaling output, region classification, and report HTML
  - `pareto_functions.py` - Shared graph costs, analytical Pareto algorithms, the analytical 2D null model, and 3D Pareto support
  - `research.py` - Self-contained Ariadne 0.3 research workflow for IPython/script use
- Tests: `tests/` directory
- OpenSpec specifications: `openspec/` directory