from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from reason.closed_loop import simulate_remove
from reason.invisible.geometry import equivalent_area, precompute_geometry_cache
from reason.schemas import PerceptionOutput


class SimulateRemoveAreaTest(unittest.TestCase):
    def _perception(self) -> PerceptionOutput:
        graph = nx.DiGraph()
        graph.add_edges_from([(1, 2), (1, 3)])
        graph.add_node(4)

        masks = {
            1: np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool),
            2: np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool),
            3: np.array([[0, 0, 0, 0], [0, 0, 1, 0]], dtype=bool),
            4: np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=bool),
        }
        node_info = {
            node: {"molmo_id": node, "label": str(node), "mask": masks[node]}
            for node in graph.nodes
        }
        return PerceptionOutput(
            target_molmo_id=99,
            task_type="test",
            occlusion_graph=graph,
            node_info=node_info,
            molmo_to_node={node: node for node in graph.nodes},
            depth=np.full((2, 4), 10.0),
        )

    def test_exposed_children_keep_pre_removal_equivalent_area(self) -> None:
        original = self._perception()
        after = simulate_remove(original, 1)

        self.assertIn(1, original.occlusion_graph)
        self.assertNotIn(1, after.occlusion_graph)
        self.assertEqual(after.occlusion_graph.in_degree(2), 0)
        self.assertEqual(after.occlusion_graph.in_degree(3), 0)

        cache = precompute_geometry_cache(after)
        self.assertEqual(equivalent_area(2, after, cache), 3.0)
        self.assertEqual(equivalent_area(3, after, cache), 2.5)

    def test_other_counterfactual_does_not_remove_a(self) -> None:
        original = self._perception()
        after_removing_other = simulate_remove(original, 4)

        self.assertIn(1, original.occlusion_graph)
        self.assertIn(1, after_removing_other.occlusion_graph)
        self.assertTrue(after_removing_other.occlusion_graph.has_edge(1, 2))
        self.assertTrue(after_removing_other.occlusion_graph.has_edge(1, 3))


if __name__ == "__main__":
    unittest.main()
