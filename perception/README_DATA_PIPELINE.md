# SmartGrasp perception

This document explains how the local `data/` split is connected to the Molmo point annotator and the occlusion graph generator.

## Overview

The pipeline connects three existing parts of the repository:

1. `data/`
   - Reads scene images, language annotations, target ids, depth maps, and instance masks.
   - Parquet files provide RGB images and language queries.
   - `npz_file.zip` provides `depth`, `instances_objects`, `instances_semantic`, `occlusion`, and `occlusion_objects`.

2. `perception/molmo/`
   - Runs `allenai/Molmo-7B-D-0924` on a scene image.
   - Produces `molmo_points.json`, where each object has `molmo_id`, `x`, `y`, and `label`.
   - Also writes `label_1_molmo.png` for visual inspection.

3. `perception/occul_map/`
   - Uses Molmo points as SAM prompts.
   - Uses SAM masks plus the scene depth map to infer occlusion edges.
   - Writes `occlusion_graph.json` and `occlusion_graph.png`.

The main entry point is:

```bash
perception/perception.py
```

## Data Flow

For a given `sceneId`, the pipeline does the following:

1. Load one matching row from the parquet files.
2. Save the RGB image to the run directory as `scene_image.png`.
3. Find the matching `{sceneId}.npz` inside `data/npz_file.zip`.
4. Save the depth map as `depth.npy`.
5. Generate points:
   - `--point-source molmo`: run Molmo on the RGB image.
   - `--point-source gt-centers`: use GT instance mask centroids as Molmo-format points.
6. Generate masks:
   - Molmo mode: use SAM with Molmo points.
   - GT mode: use `instances_objects` directly.
7. Build the occlusion graph:
   - Detect object contact regions from masks.
   - Compare depth medians in contact regions.
   - Smaller depth means closer to the camera, so that object occludes the other one.
8. Save JSON and PNG outputs.

## Default Molmo Prompt

The default prompt is designed to improve recall for small, overlapping, and partially visible objects:

```text
Point out every physically separate visible graspable object in the image. Return exactly one point for each object instance, including small, thin, overlapping, or partially hidden objects. Do not merge adjacent objects into one point, even if they touch or have similar colors. Use one point near the center of the visible region of each object. Use short noun labels. Before finishing, check the image again for any missed partially visible objects. The requested target is: {annotation}.
```

You can override it with `--prompt`.

## Recommended Command

Use the `smartgrasp` conda environment:

```bash
/home/admin128/anaconda3/envs/smartgrasp/bin/python -u perception/perception.py \
  --scene-id 527 \
  --point-source molmo \
  --sam-model-id facebook/sam-vit-large \
  --epsilon 0.01 \
  --kernel-size 5 \
  --min-contact-pixels 20 \
  --min-contact-ratio 0.001 \
  --sam-point-grid-radius 10 \
  --mask-clean-kernel 3 \
  --device cpu
```

`facebook/sam-vit-large` is slower than `facebook/sam-vit-base`, but it produced much better masks on objects that were incomplete with the base model.

## Faster Repeated Runs

Starting a new Python process reloads Molmo and SAM, which is slow. Use one of these modes when running multiple scenes.

### Batch Mode

Runs several scenes in one process. Molmo and SAM are loaded once and reused:

```bash
/home/admin128/anaconda3/envs/smartgrasp/bin/python -u perception/perception.py \
  --scene-ids 0 527 7163 \
  --point-source molmo \
  --sam-model-id facebook/sam-vit-large \
  --epsilon 0.01 \
  --kernel-size 5 \
  --min-contact-pixels 20 \
  --min-contact-ratio 0.001 \
  --sam-point-grid-radius 10 \
  --mask-clean-kernel 3 \
  --device cpu
```

### Worker Mode

Keeps the process alive. Enter one `sceneId` per line:

```bash
/home/admin128/anaconda3/envs/smartgrasp/bin/python -u perception/perception.py \
  --serve \
  --point-source molmo \
  --sam-model-id facebook/sam-vit-large \
  --epsilon 0.01 \
  --kernel-size 5 \
  --min-contact-pixels 20 \
  --min-contact-ratio 0.001 \
  --sam-point-grid-radius 10 \
  --mask-clean-kernel 3 \
  --device cpu
```

Then type:

```text
scene_id> 0
scene_id> 527
scene_id> q
```

## GT Mode

GT mode bypasses Molmo and SAM. It uses dataset instance masks directly and is useful for debugging or comparing with the predicted pipeline:

```bash
/home/admin128/anaconda3/envs/smartgrasp/bin/python -u perception/perception.py \
  --scene-id 527 \
  --point-source gt-centers \
  --epsilon 0.01 \
  --kernel-size 5 \
  --min-contact-pixels 20 \
  --min-contact-ratio 0.001
```

This generates a GT-style graph from `instances_objects + depth`.

## Outputs

Each run is written to:

```text
data/integrated_runs/scene_{sceneId}_query_{queryObjId}_{point-source}/
```

Typical files:

```text
scene_image.png
depth.npy
label_1_molmo.png
molmo_points.json
mask/
occlusion_graph.json
occlusion_graph.png
summary.json
```

In GT mode, `label_1_molmo.png` is not generated because Molmo is not run.

## Important Parameters

`--scene-id`

Run one scene.

`--scene-ids`

Run multiple scenes in one process.

`--serve`

Keep one process alive and enter scene ids interactively.

`--point-source`

Use `molmo` for predicted points or `gt-centers` for GT instance centroids.

`--sam-model-id`

SAM model id. Recommended: `facebook/sam-vit-large`.

`--sam-point-grid-radius`

Adds four extra positive SAM prompt points around each Molmo point. This helps incomplete masks. Example: `10`.

`--mask-clean-kernel`

Morphological cleanup kernel for SAM masks. Use `3` by default. Use `1` to disable cleanup.

`--epsilon`

Minimum depth difference needed to accept an occlusion direction.

`--kernel-size`

Dilation kernel used to find mask contact regions.

`--min-contact-pixels`

Minimum contact area in pixels.

`--min-contact-ratio`

Minimum contact area divided by the smaller object mask area. This filters weak accidental contacts.

## Occlusion Graph Semantics

In `occlusion_graph.json`, each edge means:

```text
source occludes target
```

The graph is built from mask contact and depth:

1. Dilate two object masks slightly.
2. Find their contact area.
3. Read depth values inside that contact area.
4. Compare median depths.
5. The object with smaller median depth is closer to the camera and becomes the edge source.

Useful edge fields:

```text
source_molmo_id
target_molmo_id
source_label
target_label
contact_pixels
contact_ratio
source_depth_median
target_depth_median
depth_gap
```

## Known Limitations

- Molmo can still miss objects, especially tiny, highly occluded, or visually merged instances.
- SAM point prompting can over-segment or under-segment if the point lies on an ambiguous region.
- Larger SAM models improve mask completeness but are slower.
- Predicted Molmo/SAM object ids are not guaranteed to match GT instance ids directly.
- GT mode is the best reference for checking whether a predicted graph misses edges.
