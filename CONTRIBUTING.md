# Contributing to dRoute

Thank you for your interest in contributing to dRoute! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to darri.eythorsson@ucalgary.ca.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your changes
5. Make your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Python 3.8+
- Git

### Building from Source

```bash
# Clone the repository
git clone https://github.com/DarriEy/dRoute.git
cd dRoute

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Build C++ components
mkdir build && cd build
cmake .. -DDMC_BUILD_PYTHON=ON -DDMC_BUILD_TESTS=ON
make -j4
cd ..

# Set PYTHONPATH for development
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python
```

### Running Tests

```bash
# Python tests
pytest tests/python/

# C++ tests
cd build
ctest --output-on-failure
```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new routing methods or functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance improvements**: Optimize existing code
- **Examples**: Add usage examples or tutorials

### Contribution Workflow

1. **Check existing issues**: Look for existing issues or create a new one to discuss your idea
2. **Fork and branch**: Fork the repo and create a feature branch
3. **Develop**: Make your changes following our coding standards
4. **Test**: Ensure all tests pass and add new tests for your changes
5. **Document**: Update documentation as needed
6. **Commit**: Write clear, descriptive commit messages
7. **Push**: Push your changes to your fork
8. **Pull Request**: Submit a PR to the main repository

## Coding Standards

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names
- Add docstrings to all public functions and classes

Format code with Black:
```bash
black python/ tests/
```

Check code style:
```bash
flake8 python/ tests/
```

### C++ Code

- Follow existing code style in the repository
- Use meaningful variable names
- Comment complex algorithms
- Prefer modern C++17 features
- Use const correctness
- Avoid raw pointers when possible (use smart pointers)

### Documentation

- Update README.md if adding new features
- Add docstrings to Python functions
- Add Doxygen comments to C++ functions
- Update CHANGELOG.md for significant changes

## Testing

### Writing Tests

- Add tests for all new features
- Ensure existing tests pass before submitting
- Aim for good test coverage
- Use descriptive test names

### Python Tests

Tests are located in `tests/python/`:

```python
def test_my_feature():
    """Test description."""
    import droute
    # Your test code
    assert result == expected
```

### C++ Tests

Tests are located in `tests/`:

```cpp
TEST_CASE("My feature test") {
    // Your test code
    REQUIRE(result == expected);
}
```

## Pull Request Process

1. **Update documentation**: Ensure the README.md and other docs reflect any changes
2. **Update CHANGELOG.md**: Add an entry under "Unreleased" section
3. **Pass all tests**: Ensure CI passes and all tests work locally
4. **Descriptive PR**: Write a clear PR description explaining:
   - What changes were made
   - Why the changes were necessary
   - How to test the changes
5. **Link issues**: Reference any related issues
6. **Review**: Respond to review feedback promptly
7. **Squash commits**: Consider squashing commits before merging

### PR Title Format

Use conventional commit format:
- `feat: Add new routing method`
- `fix: Resolve gradient computation bug`
- `docs: Update installation instructions`
- `test: Add tests for network topology`
- `refactor: Simplify router initialization`
- `perf: Optimize timestep computation`

## Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal code to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, compiler version
- **Additional context**: Error messages, logs, screenshots

Use the issue template when creating bug reports.

## Suggesting Enhancements

When suggesting enhancements:

- **Use case**: Explain the problem this would solve
- **Proposed solution**: Describe your proposed approach
- **Alternatives**: Consider alternative solutions
- **Additional context**: Any relevant examples or references

## Development Tips

### Common Tasks

**Run specific test:**
```bash
pytest tests/python/test_network.py::test_network_creation
```

**Run with coverage:**
```bash
pytest --cov=droute --cov-report=html tests/python/
```

**Build documentation (if available):**
```bash
cd docs
make html
```

**Check for memory leaks (C++):**
```bash
valgrind --leak-check=full ./build/test_executable
```

## Questions?

If you have questions about contributing:

- Check existing issues and discussions
- Open a new issue with the "question" label
- Email: darri.eythorsson@ucalgary.ca

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- GitHub contributors page
- Publication acknowledgments (for major contributions)

Thank you for contributing to dRoute! 🌊
