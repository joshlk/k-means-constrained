# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#
# License: BSD 3 clause

# NOTE on threading: these functions are stateless (no module-level mutable
# state) and their main loops run inside `with nogil` blocks, so on a
# standard CPython build multiple threads can execute them concurrently,
# and on a free-threaded (no-GIL) build they run fully in parallel.
# Callers must not mutate the input arrays from another thread while a
# call is in progress.

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating

from k_means_constrained.sklearn_import.utils.sparsefuncs_fast import assign_rows_csr

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT


np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _centers_dense(floating[:, :] X, INT[::1] labels, int n_clusters,
                   floating[:] distances):
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
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t i, j
    cdef INT c

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64
    centers_arr = np.zeros((n_clusters, n_features), dtype=dtype)
    cdef floating[:, ::1] centers = centers_arr

    n_samples_in_cluster = np.bincount(np.asarray(labels), minlength=n_clusters)
    empty_clusters = np.where(n_samples_in_cluster == 0)[0]
    # maybe also relocate small clusters?

    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = np.asarray(distances).argsort()[::-1]
        X_arr = np.asarray(X)

        for i, cluster_id in enumerate(empty_clusters):
            # XXX two relocated clusters could be close to each other
            centers_arr[cluster_id] = X_arr[far_from_centers[i]]
            n_samples_in_cluster[cluster_id] = 1

    with nogil:
        for i in range(n_samples):
            c = labels[i]
            for j in range(n_features):
                centers[c, j] += X[i, j]

    centers_arr /= n_samples_in_cluster[:, np.newaxis]

    return centers_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _centers_sparse(X, INT[::1] labels, n_clusters,
                    floating[:] distances):
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
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_samples = labels.shape[0]
    cdef Py_ssize_t i, ind, j
    cdef INT curr_label

    cdef floating[::1] data = X.data
    cdef int[::1] indices = X.indices
    cdef int[::1] indptr = X.indptr

    cdef np.ndarray[np.npy_intp, ndim=1] far_from_centers
    cdef np.ndarray[np.npy_intp, ndim=1, mode="c"] n_samples_in_cluster = \
        np.bincount(np.asarray(labels), minlength=n_clusters)
    cdef np.ndarray[np.npy_intp, ndim=1, mode="c"] empty_clusters = \
        np.where(n_samples_in_cluster == 0)[0]
    cdef int n_empty_clusters = empty_clusters.shape[0]

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64
    centers_arr = np.zeros((n_clusters, n_features), dtype=dtype)
    cdef floating[:, ::1] centers = centers_arr

    # maybe also relocate small clusters?

    if n_empty_clusters > 0:
        # find points to reassign empty clusters to
        far_from_centers = np.asarray(distances).argsort()[::-1][:n_empty_clusters]

        # XXX two relocated clusters could be close to each other
        assign_rows_csr(X, far_from_centers, empty_clusters, centers_arr)

        for i in range(n_empty_clusters):
            n_samples_in_cluster[empty_clusters[i]] = 1

    with nogil:
        for i in range(n_samples):
            curr_label = labels[i]
            for ind in range(indptr[i], indptr[i + 1]):
                j = indices[ind]
                centers[curr_label, j] += data[ind]

    centers_arr /= n_samples_in_cluster[:, np.newaxis]

    return centers_arr
