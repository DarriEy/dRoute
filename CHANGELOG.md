# Changelog

All notable changes to dRoute will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-12-27

### Added
- Initial PyPI release
- Six routing methods: Muskingum-Cunge, Lag, IRF, KWT-Soft, Diffusive Wave, and Saint-Venant
- Dual AD backends: CoDiPack (tape-based) and Enzyme (source-to-source)
- Full river network topology support with tributaries and confluences
- mizuRoute compatibility for topology.nc files
- PyTorch integration for gradient-based optimization
- Python bindings via pybind11
- Comprehensive documentation and examples
- C++ test suite

### Changed
- **BREAKING**: Package renamed from `pydmc_route` to `droute`
  - Install with: `pip install droute`
  - Import with: `import droute`
  - Backwards compatibility shim included (will be deprecated in v1.0.0)
- Unified version management across setup.py, pyproject.toml, and C++ extension
- Updated author information and repository URLs
- Improved error messages in package initialization

### Fixed
- Standardized package metadata across configuration files
- Cleaned up build artifacts and improved .gitignore

### Documentation
- Added CHANGELOG.md
- Added SECURITY.md
- Updated README.md with installation and usage examples
- Added proper Python package structure

### Deprecated
- `pydmc_route` package name (use `droute` instead)
  - Backwards compatibility maintained through v0.x releases
  - Will be removed in v1.0.0

## Previous Versions

Previous development versions (pre-0.5.0) were not released to PyPI.

[0.5.0]: https://github.com/DarriEy/dRoute/releases/tag/v0.5.0
