# k_means_constrained
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

The code is based on [scikit-lean's `KMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
and implements the same API with modifications.

Ref:
1. [Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering."
    Microsoft Research, Redmond (2000): 1-8.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf)
2. [Google's SimpleMinCostFlow C++ implementation](https://github.com/google/or-tools/blob/master/ortools/graph/min_cost_flow.h)

# Installation
Requires [Google's OR-tools to be installed](https://developers.google.com/optimization/introduction/installing/binary).

Currently tested with:
* scikit-learn == 0.19.1
* ortools == 6.7.4973

As this package uses internal scikit learn methods associated with k-means
it may break with other versions of sci-kit learn. This can be addressed
in the future by importing the internal methods into this project.

# Todo:
* Documentation
* Test with sparse `X`
* Remove dependencies on internal scikit learn methods by importing them into this project