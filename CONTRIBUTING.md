# Contributing to GAMSPy

We welcome all kinds of contributions, including bug fixes, new features, documentation improvements, and test enhancements.

Please do not open pull requests for **new features** without prior discussion. Since adding a new feature to GAMSPy creates a long-term maintenance burden, it requires consensus from the GAMSPy team.


For a more detailed overview of the development workflow and tooling, please also see the
[Developer Guide](https://gamspy.readthedocs.io/en/latest/user/developer_guide.html).

---

## Getting Started

```bash
git clone git@git.gams.com:devel/gamspy.git
cd gamspy
pip install .[test,dev,doc] # or uv sync
```

The extra dependencies install tools required for testing, development, and documentation.

---

## Workflow

We follow a standard topic-branch workflow:

1. Create or find an issue describing the bug, feature, or improvement.
2. Create a branch from the development branch (develop).
   - Branch names should reference the issue number when possible  
     (e.g. `187-add-new-feature`).
3. Make your changes with clear, focused commits.
4. Run tests and formatting checks locally.
5. Open a Pull Request.
6. Address review feedback until approval.

---

## Development Setup

### Pre-commit Hooks

We use `prek` to enforce consistent formatting and quality checks.

Install the hooks with:

```bash
prek install
```

This will automatically run checks (e.g. formatting, linting) on every commit.

---

## Code Style and Quality

- Follow **PEP 8** for Python code.
- Use clear and descriptive variable and function names.
- Write docstrings (including examples) for public classes, functions, and methods.
- Keep commits small and focused.
- Prefer clarity over cleverness.

Formatting and linting are enforced via pre-commit hooks.

---

## Testing

GAMSPy uses **pytest** for testing.

Run the test suite with:

```bash
pytest
```

You can also run specific test groups using markers:

```bash
pytest -m "unit or integration or model_library"
```

Some tests require external services such as **GAMS Engine** or **NEOS**.  
Credentials for these services can be provided via environment variables or a `.env` file.

Additional solvers required by some tests can be installed using the GAMSPy CLI:

```bash
gamspy install solver <solver_name1> <solver_name2>
```

Please ensure that new features and bug fixes include appropriate test coverage.

---

## Documentation

Documentation is built using Sphinx.

To build the documentation locally:

```bash
cd docs
make html
```

The generated HTML files will be located in:

```
docs/_build/html
```

When changing functionality or adding new features, please update the documentation accordingly.

---

## Releases

Releases can only be performed by GAMSPy team members.

To prepare a new release:

1. Ensure all changes are complete and tested.
2. Update the version in `pyproject.toml`.
3. Update the version test:
   - `tests/test_gamspy.py::test_version`
4. Add release notes under:
   - `docs/release/release_<version>.rst`
5. Update the documentation version selector (`switcher.json`).
6. Generate the changelog using `towncrier`:

```bash
towncrier build --yes --version <new_version>
```

You can automate most of these steps with:

```bash
python scripts/update_version.py <new_version>
```

Merging the release branch into `master` triggers the publishing workflow, including building and uploading wheels to PyPI and updating documentation.

---

## Community and Support

- Use the issues section for bug reports and feature requests.
- Ask questions or start discussions via issues section.

---

Thank you for contributing to **GAMSPy**!
