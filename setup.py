
"""
To build pip tar:
`python setup.py sdist`

To compile cython (better to use the compile_cython.py script):
`python setup.py build_ext`

Based on template: https://github.com/pypa/sampleproject/blob/master/setup.py
"""

from setuptools import setup, find_packages
from pip.req import parse_requirements
from codecs import open # To use a consistent encoding
from os import path
from Cython.Build import cythonize
import numpy as np

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = parse_requirements('requirements.txt', session=False)

# Profile cython and output html with annotation
cython_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(
    name='k_means_constrained',
    version='0.3.0',
    description='K-Means clustering constrained with minimum and maximum cluster size',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/joshlk/k-mean-constrained',
    author='Josh Levy-Kramer',
    keywords='kmeans k-means minimum maximum cluster segmentation size',
    packages=find_packages(),
    install_requires=[str(e.req) for e in install_requires],

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # For cython
    ext_modules=cythonize("k_means_constrained/mincostflow_vectorized_.pyx", **cython_options),
    include_dirs=[np.get_include()]
)