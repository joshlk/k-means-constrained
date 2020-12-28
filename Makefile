.PHONY: build dist redist install dist-no-cython install-from-source clean uninstall venv-create venv-activate check-dist test-pypi pypi-upload

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

venv-delete:
	rm -rf k-means-env

docs:
	sphinx-build -b html docs_source docs

download-dists:
	# e.g. `make download-dists ID=8`
	# ID is build id. You can see this in the URL when you click into the Azure pipeline
	# Need jq installed. `brew install jq`
	rm -r dist
	mkdir dist
	curl -s "https://dev.azure.com/josh0282/k-means-constrained/_apis/build/builds/$(ID)/artifacts" \
		| jq -r '.value[].resource.downloadUrl' \
		| wget --content-disposition -P dist -i -
	unzip -j -o -d dist dist/\*.zip
	rm dist/*.zip
	rm dist/*linux*.whl  # Exclude Linux wheels as would need to build manylinux which is difficult. Linux is ok to compile from source

check-dist:
	twine check dist/*

test-pypi:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

pypi-upload:
	twine upload dist/*