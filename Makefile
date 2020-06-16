.PHONY: build dist redist install install-from-source clean uninstall venv-create venv-activate

build:
	CYTHONIZE=1 ./setup.py build

dist:
	CYTHONIZE=1 ./setup.py sdist bdist_wheel

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

venv-activate:
	# Doesn't work. Need to execute manually
	source k-means-env/bin/activate