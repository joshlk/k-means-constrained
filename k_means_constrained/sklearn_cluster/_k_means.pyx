# cython: profile=True
# Profiling is enabled by default as the overhead does not seem to be measurable
# on this specific use case.

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#
# License: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating

from sklearn.utils.sparsefuncs_fast import assign_rows_csr

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

ctypedef floating (*DOT)(int N, floating *X, int incX, floating *Y,
                         int incY)


np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _centers_dense(np.ndarray[floating, ndim=2] X,
        np.ndarray[INT, ndim=1] labels, int n_clusters,
        np.ndarray[floating, ndim=1] distances):
    """M step of the K-means EM algorithm

    Computation of cluster centers / means.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    labels : array of integers, shape (n_samples)
        Current label assignment

    n_clusters : int
        Number of desired clusters

    distances : array-like, shape (n_samples)
        Distance to closest cluster for each sample.

    Returns
    -------
    centers : array, shape (n_clusters, n_features)
        The resulting centers
    """
    ## TODO: add support for CSR input
    cdef int n_samples, n_features
    n_samples = X.shape[0]
    n_features = X.shape[1]
    cdef int i, j, c
    cdef np.ndarray[floating, ndim=2] centers
    if floating is float:
        centers = np.zeros((n_clusters, n_features), dtype=np.float32)
    else:
        centers = np.zeros((n_clusters, n_features), dtype=np.float64)

    n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
    empty_clusters = np.where(n_samples_in_cluster == 0)[0]
    # maybe also relocate small clusters?

    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            # XXX two relocated clusters could be close to each other
            new_center = X[far_from_centers[i]]
            centers[cluster_id] = new_center
            n_samples_in_cluster[cluster_id] = 1

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]

    centers /= n_samples_in_cluster[:, np.newaxis]

    return centers


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _centers_sparse(X, np.ndarray[INT, ndim=1] labels, n_clusters,
        np.ndarray[floating, ndim=1] distances):
    """M step of the K-means EM algorithm

    Computation of cluster centers / means.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix, shape (n_samples, n_features)

    labels : array of integers, shape (n_samples)
        Current label assignment

    n_clusters : int
        Number of desired clusters

    distances : array-like, shape (n_samples)
        Distance to closest cluster for each sample.

    Returns
    -------
    centers : array, shape (n_clusters, n_features)
        The resulting centers
    """
    cdef int n_features = X.shape[1]
    cdef int curr_label

    cdef np.ndarray[floating, ndim=1] data = X.data
    cdef np.ndarray[int, ndim=1] indices = X.indices
    cdef np.ndarray[int, ndim=1] indptr = X.indptr

    cdef np.ndarray[floating, ndim=2, mode="c"] centers
    cdef np.ndarray[np.npy_intp, ndim=1] far_from_centers
    cdef np.ndarray[np.npy_intp, ndim=1, mode="c"] n_samples_in_cluster = \
        np.bincount(labels, minlength=n_clusters)
    cdef np.ndarray[np.npy_intp, ndim=1, mode="c"] empty_clusters = \
        np.where(n_samples_in_cluster == 0)[0]
    cdef int n_empty_clusters = empty_clusters.shape[0]

    if floating is float:
        centers = np.zeros((n_clusters, n_features), dtype=np.float32)
    else:
        centers = np.zeros((n_clusters, n_features), dtype=np.float64)

    # maybe also relocate small clusters?

    if n_empty_clusters > 0:
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1][:n_empty_clusters]

        # XXX two relocated clusters could be close to each other
        assign_rows_csr(X, far_from_centers, empty_clusters, centers)

        for i in range(n_empty_clusters):
            n_samples_in_cluster[empty_clusters[i]] = 1

    for i in range(labels.shape[0]):
        curr_label = labels[i]
        for ind in range(indptr[i], indptr[i + 1]):
            j = indices[ind]
            centers[curr_label, j] += data[ind]

    centers /= n_samples_in_cluster[:, np.newaxis]

    return centers