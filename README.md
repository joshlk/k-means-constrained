# k_means_constrained
K-means clustering problem with a minimum and/or maximum size for each cluster constraint.

The constrained assignment is formulated as a Minimum Cost Flow (MCF) linear network optimisation
problem. This is then solved using a cost-scaling push-relabel algorithm. It uses
Google's Operations Research tools's `SimpleMinCostFlow` which is a fast C++ implmentation.

This implmentation is inspired by [Bradley et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf).
Additional edges are

Useful links:
1. [Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering."
    Microsoft Research, Redmond (2000): 1-8.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf)
2. [Google's SimpleMinCostFlow implementation](https://github.com/google/or-tools/blob/master/ortools/graph/min_cost_flow.h)

# Installation
Requires [Google's OR-tools to be installed](https://developers.google.com/optimization/introduction/installing/binary).

Currently been tested using:
* scikit-learn == 0.19.1
* ortools == 6.7.4973

As this package uses internal scikit learn methods assiciated with k-means
it may break with other versions of sci-kit learn. This can be addressed
in the future by importing the internal methods into this project.

# Todo:
* Remove dependencies on internal scikit learn methods by importing them into this project