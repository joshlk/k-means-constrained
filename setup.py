#!/usr/bin/env python3

"""
Based on template: https://github.com/FedericoStra/cython-package-example
"""

from setuptools import dist, find_packages

import os
from setuptools import setup, Extension

try:
    from numpy import get_include
except:
    def get_include():
        # Defer import to later
        from numpy import get_include
        return get_include()

try:
    from Cython.Build import cythonize
except ImportError:
    print("! Could not import Cython !")
    cythonize = None


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

extensions = [
    Extension("k_means_constrained.sklearn_import.cluster._k_means", ["k_means_constrained/sklearn_import/cluster/_k_means.pyx"],
              include_dirs=[get_include()]),
    Extension("k_means_constrained.sklearn_import.metrics.pairwise_fast", ["k_means_constrained/sklearn_import/metrics/pairwise_fast.pyx"],
                  include_dirs=[get_include()]),
    Extension("k_means_constrained.sklearn_import.utils.sparsefuncs_fast", ["k_means_constrained/sklearn_import/utils/sparsefuncs_fast.pyx"],
                      include_dirs=[get_include()]),
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 1))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}

    # Declare the extension modules safe to run without the GIL (PEP 703).
    # This sets the Py_mod_gil slot to Py_MOD_GIL_NOT_USED so that importing
    # them on a free-threaded CPython build (3.13t+) does not silently
    # re-enable the GIL. Requires Cython >= 3.1; on older Cython (e.g. a local
    # dev environment) the directive is skipped and builds behave as before.
    import Cython
    if tuple(int(p) for p in Cython.__version__.split(".")[:2]) >= (3, 1):
        compiler_directives["freethreading_compatible"] = True

    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    ext_modules=extensions,
    install_requires=install_requires,
)
