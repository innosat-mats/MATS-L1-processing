# Contributing

## Environment workflows

This repository supports both pip/uv and conda users.

### Option 1: uv workflow

- Install and sync development dependencies:
  - `uv sync --group dev`
- Install package in editable mode:
  - `uv pip install -e .`

Note: ruff configuration is stored in `pyproject.toml`.

### Option 2: conda workflow

- Create and activate environment:
  - `conda env create -f deps/research-conda/environment.yml`
  - `conda activate mats-l1-processing`
- Install package in editable mode:
  - `pip install -e .`

## Dependency files and ownership

Only edit source dependency files manually:

- `deps/shared/base.in`
- `deps/infra-cdk/requirements.in`
- `deps/lambda-runtime/requirements.in`
- `deps/research-conda/environment.yml`
- `l1b_lambda/requirements.in`
- `l1b_lambda/level1b/requirements.in`

Generated lock files should not be hand-edited:

- `l1b_lambda/requirements.txt`
- `l1b_lambda/level1b/requirements.txt`

## Updating lock files

Regenerate lock files with uv:

- `uv pip compile -p 3.11 -o l1b_lambda/requirements.txt l1b_lambda/requirements.in`
- `uv pip compile -p 3.11 -o l1b_lambda/level1b/requirements.txt l1b_lambda/level1b/requirements.in`

## Lambda compatibility policy

Lambda dependencies must be available as binary wheels for CPython 3.11.

CI/CD validates both architectures:

- `manylinux2014_x86_64`
- `manylinux2014_aarch64`

If a lock update fails wheel validation, pin a compatible package version and regenerate locks.
