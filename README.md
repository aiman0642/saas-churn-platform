# \# ml-project-template

# 

# !\[CI](https://github.com/aiman0642/ml-project-template/actions/workflows/ci.yml/badge.svg)

# 

# A reusable starting point for Python ML projects: tested, linted, type-checked, and checked by CI on every push.

# 

# \## What's included

# 

# \- `src/` layout with a proper installable package (`pyproject.toml`)

# \- `pytest` tests

# \- `ruff` for linting and formatting, `mypy` in strict mode

# \- `pre-commit` hooks that block badly formatted or untyped code

# \- GitHub Actions CI that runs lint, format check, type check, and tests

# 

# \## Quick start

# 

# ```bash

# python -m venv .venv

# .venv\\Scripts\\activate          # Windows (cmd). Mac/Linux: source .venv/bin/activate

# pip install -e ".\[dev]"

# pre-commit install

# ```

# 

# \## Run checks locally

# 

# ```bash

# ruff check . \&\& ruff format --check . \&\& mypy \&\& pytest -q

# ```

# 

# \## Using this as a template

# 

# Click \*\*Use this template\*\* on GitHub, then rename `src/mltemplate` and the `name` field in `pyproject.toml`.

