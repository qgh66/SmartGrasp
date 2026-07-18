"""Factory for selecting a grasp-execution gripper without changing defaults."""

from .robot_gripper import JakaZu3Robotiq85Gripper


GRIPPER_MODELS = ("robotiq85", "box_parallel")


def create_gripper(model: str = "robotiq85", **kwargs):
    if model == "robotiq85":
        return JakaZu3Robotiq85Gripper(**kwargs)
    if model == "box_parallel":
        from .box_parallel_gripper import JakaZu3BoxParallelGripper

        return JakaZu3BoxParallelGripper(**kwargs)
    raise ValueError(f"Unknown gripper model {model!r}; choose from {GRIPPER_MODELS}")
