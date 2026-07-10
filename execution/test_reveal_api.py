import unittest

import numpy as np

from reveal_api import execute_reveal_action

VERTICAL_PUSH_ROTATION = np.column_stack((
    np.array([0.0, 0.0, -1.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
))


class RevealApiTest(unittest.TestCase):
    def test_push_plan_moves_only_along_positive_x(self):
        result = execute_reveal_action(
            occluder_id=4,
            center_point=[0.20, 0.30, 0.04],
            action_type="push",
            move_distance=0.05,
        )

        np.testing.assert_allclose(
            result["start_translation"], [0.20, 0.30, 0.04])
        np.testing.assert_allclose(
            result["push_vector"], [0.05, 0.0, 0.0])
        np.testing.assert_allclose(
            result["new_translation"], [0.25, 0.30, 0.04])
        np.testing.assert_allclose(
            result["default_rotation"], VERTICAL_PUSH_ROTATION)
        self.assertAlmostEqual(np.linalg.det(result["default_rotation"]), 1.0)
        self.assertEqual(result["action_executed"], "push")
        self.assertAlmostEqual(result["move_distance"], 0.05)
        self.assertIs(result["request_reloop"], True)

    def test_reveal_plan_rejects_invalid_inputs(self):
        invalid_cases = [
            ([0.2, 0.3], "push", 0.05),
            ([0.2, 0.3, 0.04], "unsupported", 0.05),
            ([0.2, 0.3, 0.04], "push", 0.0),
            ([0.2, 0.3, 0.04], "push", -0.05),
        ]
        for center_point, action_type, move_distance in invalid_cases:
            with self.subTest(
                    center_point=center_point,
                    action_type=action_type,
                    move_distance=move_distance):
                with self.assertRaises(ValueError):
                    execute_reveal_action(
                        occluder_id=4,
                        center_point=center_point,
                        action_type=action_type,
                        move_distance=move_distance,
                    )


if __name__ == "__main__":
    unittest.main()
