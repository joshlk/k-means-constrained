[![PyPI](https://img.shields.io/pypi/v/k-means-constrained)](https://pypi.org/project/k-means-constrained/)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![Build](https://github.com/joshlk/k-means-constrained/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/joshlk/k-means-constrained/actions/workflows/build_wheels.yml)
[**Documentation**](https://joshlk.github.io/k-means-constrained/)

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

It is supported on Python 3.10, 3.11, 3.12 and 3.13. Previous versions of k-means-constrained support older versions of Python and Numpy.

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

<details>
  <summary>Code only</summary>
    
```
from k_means_constrained import KMeansConstrained
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
                [4, 2], [4, 4], [4, 0]])
clf = KMeansConstrained(
     n_clusters=2,
     size_min=2,
     size_max=5,
     random_state=0
 )
clf.fit_predict(X)
clf.cluster_centers_
clf.labels_
```
    
</details>

# Time complexity and runtime

k-means-constrained is a more complex algorithm than vanilla k-means and therefore will take longer to execute and has worse scaling characteristics.

Given a number of data points $n$ and clusters $c$, the time complexity of:
* k-means: $\mathcal{O}(nc)$
* k-means-constrained<sup>1</sup>: $\mathcal{O}((n^3c+n^2c^2+nc^3)\log(n+c)))$

This assumes a constant number of algorithm iterations and data-point features/dimensions.

If you consider the case where $n$ is the same order as $c$ ($n \backsim c$) then:
* k-means: $\mathcal{O}(n^2)$
* k-means-constrained<sup>1</sup>: $\mathcal{O}(n^4\log(n)))$

Below is a runtime comparison between k-means and k-means-constrained whereby the number of iterations, initializations, multi-process pool size and dimension size are fixed. The number of clusters is also always one-tenth the number of data points $n=10c$. It is shown above that the runtime is independent of the minimum or maximum cluster size, and so none is included below.

<p align="center">
<img src="https://raw.githubusercontent.com/joshlk/k-means-constrained/master/etc/execution_time.png" alt="Data-points vs execution time for k-means vs k-means-constrained. Data-points=10*clusters. No min/max constraints" width="50%" height="50%">
</p>

<details>
  <summary>System details</summary>
    
* OS: Linux-5.15.0-75-generic-x86_64-with-glibc2.35
* CPU: AMD EPYC 7763 64-Core Processor
* CPU cores: 120
* k-means-constrained version: 0.7.3
* numpy version: 1.24.2
* scipy version: 1.11.1
* ortools version: 9.6.2534
* joblib version: 1.3.1
* sklearn version: 1.3.0
</details>
---

<sup>1</sup>: [Ortools states](https://developers.google.com/optimization/reference/graph/min_cost_flow) the time complexity of their cost-scaling push-relabel algorithm for the min-cost flow problem as $\mathcal{O}(n^2m\log(nC))$ where $n$ is the number of nodes, $m$ is the number of edges and $C$ is the maximum absolute edge cost.

# Change log

* v0.7.6 (2025-06-30) Add Python v3.13 and Linux ARM support.
* v0.7.5 fix comment in README on Python version that is supported
* v0.7.4 compatible with Numpy +v2.1.1. Added Python 3.12 support and dropped Python 3.8 and 3.9 support (due to Numpy). Linux ARM support has been dropped as we use GitHub runners to build the package and ARM machines was being emulated using QEMU. This however was producing numerical errors. GitHub should natively support Ubuntu ARM images soon and then we can start to re-build them.
* v0.7.3 compatible with Numpy v1.23.0 to 1.26.4

# Citations

If you use this software in your research, please use the following citation:

```
@software{Levy-Kramer_k-means-constrained_2018,
  author = {Levy-Kramer, Josh},
  month = apr,
  title = {{k-means-constrained}},
  url = {https://github.com/joshlk/k-means-constrained},
  year = {2018}
}
```
