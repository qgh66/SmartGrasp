#!/usr/bin/env python
"""Render one reproducible RGB-D frame from a compact stacked layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from simulation.camera import VirtualCamera
from simulation.capture_artifacts import export_camera_frame
from simulation.scene import SimulationScene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture RGB-D and masks for a compact stacked PyBullet scene"
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "config"
            / "industrial_scene_compact_stacked_layout.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "compact_stacked_camera_preview",
    )
    return parser.parse_args()


def load_base_scene_config(config_path: Path) -> dict[str, Any]:
    resolved_config_path = config_path.expanduser().resolve()
    with resolved_config_path.open("r", encoding="utf-8") as config_file:
        scene_config = json.load(config_file)

    object_specs = scene_config.get("objects", [])
    if not object_specs:
        raise ValueError(f"Scene config has no objects: {resolved_config_path}")

    resolved_objects = []
    for original_spec in object_specs:
        object_spec = dict(original_spec)
        mesh_path = Path(str(object_spec["path"])).expanduser()
        if not mesh_path.is_absolute():
            workspace_candidate = WORKSPACE_ROOT / mesh_path
            config_candidate = resolved_config_path.parent / mesh_path
            mesh_path = (
                workspace_candidate
                if workspace_candidate.exists()
                else config_candidate
            )
        object_spec["path"] = str(mesh_path.resolve())
        resolved_objects.append(object_spec)

    scene_config["_path"] = str(resolved_config_path)
    scene_config["_resolved_objects"] = resolved_objects
    return scene_config


def load_layout(layout_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_layout_path = layout_path.expanduser().resolve()
    with resolved_layout_path.open("r", encoding="utf-8") as layout_file:
        layout = json.load(layout_file)

    base_scene_path = Path(layout["base_scene_config"])
    if not base_scene_path.is_absolute():
        base_scene_path = resolved_layout_path.parent / base_scene_path
    scene_config = load_base_scene_config(base_scene_path)

    object_overrides = layout.get("object_overrides", {})
    configured_names = {
        str(spec.get("name")) for spec in scene_config["_resolved_objects"]
    }
    unknown_names = sorted(set(object_overrides) - configured_names)
    if unknown_names:
        raise ValueError(
            "Compact layout references unknown object(s): " + ", ".join(unknown_names)
        )

    resolved_objects = []
    for original_spec in scene_config["_resolved_objects"]:
        object_spec = dict(original_spec)
        object_spec.update(object_overrides.get(str(object_spec.get("name")), {}))
        resolved_objects.append(object_spec)
    scene_config["_resolved_objects"] = resolved_objects
    scene_config["camera"] = dict(layout.get("camera", scene_config.get("camera", {})))
    scene_config["_compact_layout_path"] = str(resolved_layout_path)
    return layout, scene_config


def capture_scene(layout: dict[str, Any], scene_config: dict[str, Any], output_dir: Path) -> None:
    scene = SimulationScene(gui=False)
    scene.connect()
    try:
        scene.load_plane()
        scene.load_objects(scene_config["_resolved_objects"])
        # Lock the intentionally dense layout before gravity can separate it.
        scene.stage_objects_at_initial_poses()
        scene.step(10)

        camera_config = scene_config["camera"]
        camera = VirtualCamera(
            position=tuple(camera_config["position"]),
            target=tuple(camera_config["target"]),
            near=float(camera_config.get("near", 0.01)),
            far=float(camera_config.get("far", 3.0)),
            width=int(camera_config.get("width", 1280)),
            height=int(camera_config.get("height", 720)),
            fov=float(camera_config.get("fov", 45.0)),
        )
        rgb, depth, segmentation = camera.capture()
        target_body_id = scene.get_body_id_by_name(layout["target_object"])
        metadata = export_camera_frame(
            output_dir=output_dir,
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            object_names_by_id={
                int(body_id): scene.get_object_info(body_id).name
                for body_id in scene.object_ids
            },
            target_body_id=target_body_id,
        )

        capture_metadata_path = output_dir / "capture.json"
        metadata.update(
            {
                "source_scene_config": scene_config["_path"],
                "compact_layout": scene_config["_compact_layout_path"],
                "target_object_name": layout["target_object"],
                "object_count": len(scene.object_ids),
                "camera": camera_config,
            }
        )
        capture_metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Compact stacked RGB-D capture: {output_dir.resolve()}")
        print(
            "Depth range: "
            f"{metadata['depth_min_m']:.6f}–{metadata['depth_max_m']:.6f} m"
        )
        print(
            "Visible configured objects: "
            f"{sum(item['pixel_count'] > 0 for item in metadata['object_masks'].values())}"
            f"/{len(scene.object_ids)}"
        )
    finally:
        scene.disconnect()


def main() -> None:
    args = parse_args()
    layout, scene_config = load_layout(args.layout)
    capture_scene(layout, scene_config, args.output_dir)


if __name__ == "__main__":
    main()
