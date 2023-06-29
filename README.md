[![PyPI](https://img.shields.io/pypi/v/k-means-constrained)](https://pypi.org/project/k-means-constrained/)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![Build Status](https://dev.azure.com/josh0282/k-means-constrained/_apis/build/status/joshlk.k-means-constrained?branchName=master)](https://dev.azure.com/josh0282/k-means-constrained/_build/latest?definitionId=1&branchName=master)
[![Documentation](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat)](https://joshlk.github.io/k-means-constrained/)

# k-means-constrained
K-means clustering implementation whereby a minimum and/or maximum size for each
cluster can be specified.

This K-means implementation modifies the cluster assignment step (E in EM)
by formulating it as a Minimum Cost Flow (MCF) linear network
optimisation problem. This is then solved using a cost-scaling
push-relabel algorithm and uses [Google's Operations Research tools's
`SimpleMinCostFlow`](https://developers.google.com/optimization/flow/mincostflow)
which is a fast C++ implementation.

This package is inspired by [Bradley et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf).
The original Minimum Cost Flow (MCF) network proposed by Bradley et al.
has been modified so maximum cluster sizes can also be specified along
with minimum cluster size. 

The code is based on [scikit-lean's `KMeans`](https://scikit-learn.org/0.19/modules/generated/sklearn.cluster.KMeans.html)
and implements the same [API with modifications](https://joshlk.github.io/k-means-constrained/).

Ref:
1. [Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering."
    Microsoft Research, Redmond (2000): 1-8.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf)
2. [Google's SimpleMinCostFlow C++ implementation](https://github.com/google/or-tools/blob/master/ortools/graph/min_cost_flow.h)

# Installation
You can install the k-means-constrained from PyPI:

```
pip install k-means-constrained
```

It is supported on Python 3.8 and above.

# Example

More details can be found in the [API documentation](https://joshlk.github.io/k-means-constrained/).

```python
>>> from k_means_constrained import KMeansConstrained
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...                [4, 2], [4, 4], [4, 0]])
>>> clf = KMeansConstrained(
...     n_clusters=2,
...     size_min=2,
...     size_max=5,
...     random_state=0
... )
>>> clf.fit_predict(X)
array([0, 0, 0, 1, 1, 1], dtype=int32)
>>> clf.cluster_centers_
array([[ 1.,  2.],
       [ 4.,  2.]])
>>> clf.labels_
array([0, 0, 0, 1, 1, 1], dtype=int32)
```

