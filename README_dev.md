# Build and test

Notes:
* Numpy build version is in `pyproject.toml` while the runtime version is in `requirements.txt`
* Check which Python versions a new version of Numpy is compatible with. Also check ortools as this is slower to update.
* Change the Python versions in the GitHub action and the badge in the README
* You might need to increase the ciwheelbuild version in the GitHub action to be able to use new Python versions
* Check ciwheelbuild example if you need to change runner image versions (e.g. MacOS, Windows or Ubuntu):
    https://github.com/pypa/cibuildwheel/blob/main/examples/github-with-qemu.yml
* Add changes to the change log

Steps:

1. Build and test locally:

To build Cython extensions in source:
```shell script
make compile
```

To test:
```shell script
pytest
```

2. Push changes to GitHub to build it for all platforms (if you get errors check notes above)
    MacOS isn't tested so download a copy and test locally.

3. Add changes to change log and bump version  (major, minor or patch):

```shell script
bump2version patch
```



Push to GitHub to build for many version. The MacOS ARM build isn't automatically tested and so should be tested locally.

# Push to PyPi
Requires: `pip install twine`
Don't forget to increment version number

Bump version (major, minor or patch):



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
