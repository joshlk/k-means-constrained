#!/usr/bin/env python

from ortools.graph.pywrapgraph import SimpleMinCostFlow

# Cython paths must be fully qualified
from k_means_constrained.mincostflow_vectorized_ import \
    SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized, \
    SimpleMinCostFlow_SetNodeSupplyVectorized, \
    SimpleMinCostFlow_FlowVectorized


class SimpleMinCostFlowVectorized(SimpleMinCostFlow):

    def AddArcWithCapacityAndUnitCostVectorized(self, tail, head, capacity, unit_cost):
        return SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized(self, tail, head, capacity, unit_cost)

    def SetNodeSupplyVectorized(self, node, supply):
        return SimpleMinCostFlow_SetNodeSupplyVectorized(self, node, supply)

    def FlowVectorized(self, arc):
        return SimpleMinCostFlow_FlowVectorized(self, arc)