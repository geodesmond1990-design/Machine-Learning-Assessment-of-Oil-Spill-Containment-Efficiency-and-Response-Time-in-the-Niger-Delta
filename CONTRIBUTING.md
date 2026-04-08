# Contributing to Niger Delta Oil Spill ML

Thank you for your interest in contributing to this project.

## How to Contribute

### Reporting Bugs
Open a GitHub Issue with:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS

### Suggesting Enhancements
Open a GitHub Issue labelled `enhancement` with a clear description of the proposed change and its scientific motivation.

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear commit messages
4. Add or update tests in `tests/` if applicable
5. Run tests: `pytest tests/`
6. Submit a pull request describing your changes

## Code Style

- Follow PEP 8
- Use descriptive variable names
- Add docstrings to all functions (NumPy docstring style)
- Keep functions focused: one function, one responsibility

## Scientific Standards

- Any changes to ML models, PHRI weights, or statistical methods must be accompanied by justification in the PR description
- Results reproducibility must be maintained: all random seeds set to 42
- No data files should be committed (see .gitignore)

## Contact

For questions about the research methodology: [email@institution.edu.ng]
