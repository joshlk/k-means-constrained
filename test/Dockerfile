FROM python:3.7

RUN pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple k-means-constrained

RUN python -c "from k_means_constrained import KMeansConstrained; import numpy as np; clf=KMeansConstrained(1,1,1);clf.fit(np.array([[0,0]]));"
