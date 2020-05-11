
### --- Build and push to PyPi
# Requires: `pip install twine`
# Don't forget to increment version number

# Activate virtual env (see `Create virtual env` below)
source k-means-env/bin/activate

# Build distribution
python setup.py build_ext
python setup.py sdist bdist_wheel

# Check packages
twine check dist/*

# Test upload (build the source for you)
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Test install (in virtual env):
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple k-means-constrained

# Then push to real PyPI:
twine upload dist/*


### --- Create virtual env
python -m venv k-means-env
pip install wheel cython
pip install -r requirements.txt