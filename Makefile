.PHONY: build dist redist install dist-no-cython install-from-source clean venv-create venv-activate check-dist test-pypi pypi-upload

build:
	python setup.py build

dist:
	python setup.py build bdist_wheel sdist

dist-no-cython:
	CYTHONIZE=0 python setup.py build bdist_wheel

compile:
	python setup.py build build_ext --inplace

redist: clean dist

install:
	pip install .

install-from-source: dist
	pip install dist/k-means-constrained-0.5.0.tar.gz

clean:
	$(RM) -r build dist src/*.egg-info artifact
	$(RM) -r .pytest_cache
	find . -name __pycache__ -exec rm -r {} +
	#git clean -fdX

venv-create:
	conda create -n k-means-constrained python=3.10
	conda activate k-means-constrained
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

venv-activate:
	# Doesn't work. Need to execute manually
	conda activate k-means-constrained

venv-delete:
	conda env delete k-means-constrained

docs:
	sphinx-build -b html docs_source docs

source-dists:
	rm -r dist
	python setup.py sdist --formats=gztar

download-dists:
	# e.g. `make download-dists ID=8`
	# ID is run id (get from url. Not Job ID)
	# Need gh installed. `brew install gh`
	rm -r artifact || true
	gh run download $(ID)

check-dist:
	twine check artifact/*

test-pypi:
	twine upload --repository-url https://test.pypi.org/legacy/ artifact/*

pypi-upload:
	twine upload artifact/*