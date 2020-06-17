.PHONY: build dist redist install install-from-source clean uninstall venv-create venv-activate check-dist test-pypi pypi-upload

build:
	CYTHONIZE=1 python setup.py build

dist:
	CYTHONIZE=1 python setup.py sdist bdist_wheel

compile:
	CYTHONIZE=1 python setup.py build build_ext --inplace

redist: clean dist

install:
	CYTHONIZE=1 pip install .

install-from-source: dist
	pip install dist/k-means-constrained-0.3.3.tar.gz

clean:
	$(RM) -r build dist src/*.egg-info
	$(RM) -r .pytest_cache
	find . -name __pycache__ -exec rm -r {} +
	#git clean -fdX

uninstall:
	pip uninstall cython-package-example

venv-create:
	python -m venv k-means-env
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

venv-activate:
	# Doesn't work. Need to execute manually
	source k-means-env/bin/activate

check-dist:
	twine check dist/*

test-pypi:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*


pypi-upload:
	twine upload dist/*


