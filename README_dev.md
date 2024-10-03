# Build and test

To build cython exstentions in source:
```shell script
make compile
```

To test:
```shell script
pytest
```

Push to GitHub to build for many version. The MacOS ARM build isn't automatically tested and so should be tested locally.

# Push to PyPi
Requires: `pip install twine`
Don't forget to increment version number

Bump version (major, minor or patch):

```shell script
bump2version patch
```

Download distributions (artifacts)

```shell script
make download-dists ID=$BUILD_ID
```

Upload to test PyPi

```shell script
make check-dist
make test-pypi
```

Activate virtual env (might need to `make venv-create`)

```shell script
source k-means-env/bin/activate
```

Test install (in virtual env. *****Remember to cd out of k-means-constrained folder*****):

```shell script
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple k-means-constrained
```

Then push to real PyPI:

```shell script
make pypi-upload
```
