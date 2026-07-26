from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np


class Branch(str, Enum):
    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    FAULT = "fault"


@dataclass
class PerceptionOutput:
    """Shared input to the reasoning pipeline."""
    # Required graph fields
    target_molmo_id: Optional[int]
    task_type: str
    occlusion_graph: nx.DiGraph
    node_info: dict[int, dict]
    molmo_to_node: dict[int, int]

    # Optional scene artifacts
    depth: Optional[np.ndarray] = None
    labeled_rgb: Optional[np.ndarray] = None
    occlusion_graph_rgb: Optional[np.ndarray] = None
    occlusion_graph_path: Optional[Path] = None
    final_objects_sheet: Optional[np.ndarray] = None
    final_objects_sheet_path: Optional[Path] = None
    sam2_rgb_parts_sheet: Optional[np.ndarray] = None
    sam2_rgb_parts_sheet_path: Optional[Path] = None
    # Canonical validated object/part ownership from perception/summary.json.
    object_id_to_part_ids: Optional[dict[int, tuple[int, ...]]] = None
    part_id_to_object_id: Optional[dict[int, int]] = None

    # Backward-compatible aliases for older summaries and callers.
    object_id_to_sam2_part_ids: Optional[dict[int, tuple[int, ...]]] = None
    object_id_to_sam2_part_files: Optional[dict[int, tuple[str, ...]]] = None
    prior_prompt_mode: str = "original"
    ranking_score: str = "legacy"
    scene_id: Optional[int] = None
    annotation: Optional[str] = None
    point_source: Optional[str] = None
    output_dir: Optional[Path] = None


@dataclass
class GraspDecision:
    """Decision returned by a branch handler."""
    branch: Branch
    grasp_id: Optional[int] = None
    grasp_label: Optional[str] = None
    target_molmo_id: Optional[int] = None
    is_terminal: bool = False
    success: bool = True
    message: str = ""
    details: Optional[dict] = None   # Candidate scores {mid: {P_s, P_g, P, IG, ...}}
