FROM python:3.7.6

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY requirements_test.txt requirements_test.txt
RUN pip install -r requirements_test.txt

# You need to compile cython seperatly (using pycharm run configuration)

ENV PYTHONPATH=/