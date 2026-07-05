# CLAUDE.md

This file provides guidance for AI assistants working with the k-means-constrained codebase.

## Project Overview

**k-means-constrained** is a Python library implementing K-means clustering with minimum and/or maximum cluster size constraints. It extends scikit-learn's KMeans API by formulating the constrained assignment step (E-step) as a Minimum Cost Flow (MCF) network optimization problem, solved using Google OR-Tools' `SimpleMinCostFlow`.

- **Author:** Josh Levy-Kramer
- **License:** BSD 3-Clause
- **Version:** 0.9.0
- **Python support:** 3.10, 3.11, 3.12, 3.13, 3.14

## Repository Structure

```
k_means_constrained/                # Main package
├── __init__.py                     # Exports KMeansConstrained, defines __version__
├── k_means_constrained_.py         # Core algorithm implementation
└── sklearn_import/                 # Vendored scikit-learn code (modified)
    ├── base.py                     # BaseEstimator, ClusterMixin, TransformerMixin
    ├── exceptions.py
    ├── cluster/
    │   ├── _k_means.pyx            # Cython: M-step center computation
    │   └── k_means_.py             # KMeans base class, k-means++ init
    ├── metrics/
    │   ├── pairwise.py             # Distance computations
    │   └── pairwise_fast.pyx       # Cython: optimized pairwise distances
    ├── utils/
    │   ├── extmath.py              # row_norms, squared_norm
    │   ├── validation.py           # Input validation (check_array, etc.)
    │   └── sparsefuncs_fast.pyx    # Cython: sparse matrix operations
    └── preprocessing/
tests/
├── test_k_means_constrained_.py    # Core algorithm tests
└── test_kmeans_constrained_from_sklearn.py  # Sklearn-adapted tests
etc/                                # Benchmarks and notebooks
docs_source/                        # Sphinx documentation source
docs/                               # Built HTML documentation
.github/workflows/build_wheels.yml  # CI/CD pipeline
```

## Build & Development Commands

### Prerequisites

Requires Cython and numpy at build time. Install all dev dependencies:

```sh
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Key Commands

| Command | Purpose |
|---|---|
| `make compile` | Build Cython extensions in-place (required before running tests locally) |
| `pytest` | Run all tests |
| `pytest tests/test_k_means_constrained_.py` | Run core tests only |
| `make build` | Build the package |
| `make dist` | Build wheel and sdist |
| `make clean` | Remove build artifacts and caches |
| `make docs` | Build Sphinx HTML documentation |

### Typical Development Workflow

1. `make compile` — build Cython extensions in-place
2. Edit Python or Cython source files
3. `make compile` again if `.pyx` files were changed
4. `pytest` — run tests

## Architecture & Key Concepts

### Algorithm Flow

1. **Initialization:** k-means++ or random center selection (in `sklearn_import/cluster/k_means_.py:_k_init`)
2. **E-step (constrained):** `_labels_constrained()` builds an MCF graph from distance matrix and solves it via `ortools.SimpleMinCostFlow` to assign points to clusters respecting size_min/size_max
3. **M-step (standard):** `_centers_dense()` / `_centers_sparse()` in `_k_means.pyx` recomputes cluster centers
4. **Iterate** until convergence or max iterations

### Key Functions in `k_means_constrained_.py`

- `KMeansConstrained` — main API class, sklearn-compatible estimator
- `k_means_constrained()` — top-level function handling multiple random inits
- `kmeans_constrained_single()` — single run of the constrained E-M loop
- `_labels_constrained()` — constrained E-step using min-cost flow
- `minimum_cost_flow_problem_graph()` — builds MCF graph (nodes, arcs, costs, capacities)
- `solve_min_cost_flow_graph()` — solves the MCF problem via OR-Tools

### Vendored sklearn Code

The `sklearn_import/` directory contains code copied and adapted from scikit-learn. This is not a dependency on sklearn at runtime — it's vendored to avoid version coupling. Changes to these files should be minimal and well-documented.

## Cython Extensions

Three Cython `.pyx` files compile to C extensions:

| Extension | Source | Purpose |
|---|---|---|
| `cluster._k_means` | `_k_means.pyx` | Compute cluster centers (M-step) |
| `metrics.pairwise_fast` | `pairwise_fast.pyx` | Optimized sparse distance computation |
| `utils.sparsefuncs_fast` | `sparsefuncs_fast.pyx` | Sparse CSR row norms and stats |

Compilation is controlled by the `CYTHONIZE` environment variable (defaults to `1`). Set `CYTHONIZE=0` to skip Cythonization and use pre-compiled `.c`/`.cpp` files.

Cython compiler directives: `language_level=3`, `embedsignature=True`, `freethreading_compatible=True` (requires Cython >= 3.1). Extensions use `boundscheck(False)`, `wraparound(False)`, `cdivision(True)` for performance.

## Testing

- **Framework:** pytest
- **Test files:** `tests/test_k_means_constrained_.py` (core algorithm), `tests/test_kmeans_constrained_from_sklearn.py` (sklearn compatibility)
- **CI matrix:** Ubuntu (x64+ARM), Windows, macOS (Intel+Apple Silicon) x Python 3.10-3.14
- **CI tool:** `cibuildwheel` v3.0.0 — builds and tests wheels across platforms
- **CI triggers:** push to master, PRs, weekly schedule (Thursday 1 AM UTC), manual dispatch
- **Note:** musllinux is skipped (ortools compatibility). Free-threaded (cp314t) wheels are built and tested on Linux only (ortools has no free-threaded macOS/Windows wheels).

## Free-threading (no-GIL) support

- The Cython extensions declare `freethreading_compatible=True` and run their main loops `nogil`; concurrent use is covered by `tests/test_threading.py`.
- `ortools` does not declare `Py_mod_gil` support, so importing it on a free-threaded build re-enables the GIL with a RuntimeWarning — users must set `PYTHON_GIL=0` until fixed upstream. Its `solve()` binding also holds the GIL on standard builds, so `n_init` runs use joblib `prefer="threads"` only when `sys._is_gil_enabled()` is False; GIL builds keep the process backend.

## Dependencies

### Runtime (`requirements.txt`)

- `ortools >= 9.15.6755` — Google OR-Tools for min-cost flow
- `scipy >= 1.14.1` — sparse matrices, distance functions
- `numpy >= 2.1.1` — array operations
- `six` — Python 2/3 compatibility (legacy)
- `joblib` — parallel execution of multiple inits

### Build (`pyproject.toml`)

- `setuptools`, `wheel`, `cython >= 3.0.11`, `numpy >= 2.0, < 3`

### Dev (`requirements-dev.txt`)

- `pytest`, `pandas`, `scikit-learn >= 1.5.2`, `sphinx`, `bump2version`, `twine`

## Versioning & Release

### Version locations

The version string appears in three files that must stay in sync:

| File | Format |
|---|---|
| `setup.cfg` | `version = X.Y.Z` |
| `k_means_constrained/__init__.py` | `__version__ = 'X.Y.Z'` |
| `.bumpversion.cfg` | `current_version = X.Y.Z` |

### Bumping the version

Use `bump2version` (config in `.bumpversion.cfg`) which updates all three files and creates a git commit + tag automatically:

```sh
bump2version patch   # 0.9.0 → 0.9.1
bump2version minor   # 0.9.0 → 0.10.0
bump2version major   # 0.9.0 → 1.0.0
```

If bumping manually (without `bump2version`), update all three files listed above.

### Changelog

The changelog lives in `README.md` under the `# Change log` heading. When bumping the version, add a new entry at the top of the list following the existing format:

```
* vX.Y.Z (YYYY-MM-DD) Brief description of changes.
```

### Release workflow (from `README_dev.md`)

1. Build and test locally (`make compile && pytest`)
2. Push to GitHub (triggers CI wheel builds)
3. Add changelog entry in `README.md`, bump version (`bump2version patch|minor|major`), push again
4. Download CI artifacts (`make download-dists ID=$BUILD_ID`)
5. Upload to test PyPI (`make test-pypi`), verify install
6. Upload to real PyPI (`make pypi-upload`)

## Code Conventions

- **Naming:** PascalCase for classes, snake_case for functions, leading underscore for internal/private
- **Docstrings:** NumPy style (Parameters, Returns, Notes, Examples sections)
- **Imports:** `import numpy as np`, `import scipy.sparse as sp`
- **Type hints:** Not heavily used; Cython files use `cdef`/`ctypedef` typed declarations
- **Error handling:** `ValueError` for constraint violations, `NotImplementedError` for unsupported sparse operations
- **Style checking:** flake8 (configured in `tox.ini`, excludes `.tox`, `*.egg`, `build`, `data`)

## Important Caveats

- Sparse matrix input is not fully supported — some code paths raise `NotImplementedError`
- Performance: O(n^4 log n) when n ~ c (number of clusters), vs O(n^2) for standard k-means. Not suitable for very large datasets with few clusters.
- The `tox.ini` is outdated (references Python 3.8/3.9). Use `pytest` directly or rely on CI.
- Always run `make compile` after modifying `.pyx` files before testing.
