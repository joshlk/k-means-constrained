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

3. Add changes to change log and bump version  (major, minor or patch):

```shell script
bump2version patch
```

4. Download distributions (artifacts)

```shell script
make download-dists ID=$BUILD_ID
```

5. Upload to test PyPi

```shell script
make check-dist
make test-pypi
```

6. Activate virtual env (might need to `make venv-create`)

```shell script
source k-means-env/bin/activate
```

7. Test install (in virtual env. *****Remember to cd out of k-means-constrained folder*****):

```shell script
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple k-means-constrained
```

8. Then push to real PyPI:

```shell script
make pypi-upload
```
