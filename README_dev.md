
# Build and push to PyPi
Requires: `pip install twine`
Don't forget to increment version number

Activate virtual env (might need to `make venv-create`)

```shell script
source k-means-env/bin/activate
```

Build distribution

```shell script
make dist
```

Check packages

```shell script
make check-dist
```

Test upload (build the source for you)

```shell script
make test-pypi
```

Test install (in virtual env):

```shell script
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple k-means-constrained
```

Then push to real PyPI:

```shell script
make pypi-upload
```
