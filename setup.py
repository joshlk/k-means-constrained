#!/usr/bin/env python3

"""
Based on template: https://github.com/FedericoStra/cython-package-example
"""

from setuptools import dist
dist.Distribution().fetch_build_eggs(["cython>=0.29", "numpy>=1.13"])

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
    Extension("k_means_constrained.mincostflow_vectorized_", ["k_means_constrained/mincostflow_vectorized_.pyx"],
              include_dirs=[get_include()]),
    Extension("k_means_constrained.sklearn_cluster._k_means", ["k_means_constrained/sklearn_cluster/_k_means.pyx"],
              include_dirs=[get_include()]),
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")

setup(
    ext_modules=extensions,
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": ["sphinx", "sphinx-rtd-theme"]
    },
)
