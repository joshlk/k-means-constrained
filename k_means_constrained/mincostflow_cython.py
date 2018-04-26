#!/usr/bin/env python

from ortools.graph.pywrapgraph import SimpleMinCostFlow

# Cython paths must be fully qualified
from k_means_constrained.mincostflow_cython_ import SimpleMinCostFlow_AddArcWithCapacityAndUnitCostNumpy


class SimpleMinCostFlowNumpy(SimpleMinCostFlow):

    def AddArcWithCapacityAndUnitCostNumpy(self, tail, head, capacity, unit_cost):
        return SimpleMinCostFlow_AddArcWithCapacityAndUnitCostNumpy(self, tail, head, capacity, unit_cost)