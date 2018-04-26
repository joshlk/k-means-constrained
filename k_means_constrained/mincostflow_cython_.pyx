
import numpy as np
cimport numpy as np
cimport cython

from ortools.graph._pywrapgraph import SimpleMinCostFlow_AddArcWithCapacityAndUnitCost

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

# cdef extern from "_pywrapgraph.so":
#     void SimpleMinCostFlow_AddArcWithCapacityAndUnitCost(DTYPE tail, DTYPE head, DTYPE capacity, DTYPE unit_cost)


@cython.boundscheck(False)
@cython.wraparound(False)
def SimpleMinCostFlow_AddArcWithCapacityAndUnitCostNumpy(
        self,
        np.ndarray[DTYPE_t, ndim=1] tail, np.ndarray[DTYPE_t, ndim=1] head,
        np.ndarray[DTYPE_t, ndim=1] capacity, np.ndarray[DTYPE_t, ndim=1] unit_cost):

    cdef int len = tail.shape[0]

    assert tail.dtype == DTYPE
    assert head.dtype == DTYPE
    assert capacity.dtype == DTYPE
    assert unit_cost.dtype == DTYPE
    assert head.shape[0] == len
    assert capacity.shape[0] == len
    assert unit_cost.shape[0] == len

    for i in range(len):
        SimpleMinCostFlow_AddArcWithCapacityAndUnitCost(self, tail[i], head[i], capacity[i], unit_cost[i])


