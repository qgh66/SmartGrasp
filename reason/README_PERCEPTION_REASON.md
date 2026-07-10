# Perception -> Reason One-Shot Run

This note documents the shell entrypoint that runs perception first and then
runs reason on the same scene ids.

## Entry Point

Use `run_perception.sh` from the project root:

```bash
bash run_perception.sh 59
```

For multiple scenes:

```bash
bash run_perception.sh 59 242 691
```

By default, after perception succeeds, the script automatically runs reason on
the same scene ids. The reason step is single-step reasoning only; it does not
pass `--closed-loop`.

## Data Flow

The perception step writes:

```text
data/scene_<id>/perception/summary.json
data/scene_<id>/perception/scene_image.png
data/scene_<id>/perception/depth.npy
data/scene_<id>/perception/label_2_vlm.png
data/scene_<id>/perception/sam2_rgb_parts_sheet.png
data/scene_<id>/perception/sam2_rgb_parts/part_XXX.png
```

The reason step then reads the current scene summaries under `data/`:

```text
data/scene_<id>/perception/summary.json
```

It rebuilds the occlusion graph from `occlusion_matrix` and uses
`object_id_to_sam2_part_ids` plus `sam2_rgb_parts_sheet.png` when
`REASON_PRIOR_PROMPT=graspability`.

## Default Reason Settings

`run_perception.sh` uses these defaults for the automatic reason step:

```bash
RUN_REASON_AFTER_PERCEPTION=1
REASON_DATA_ROOT=data
REASON_OUT_ROOT=runs_reason_current
REASON_MODEL=gpt-5.5
REASON_PRIOR_PROMPT=graspability
REASON_RANKING_SCORE=ig_graspability
REASON_TARGET_SOURCE=auto
```

Equivalent reason command for a single scene:

```bash
python test.py \
  --root data \
  --scene-id 59 \
  --target-source auto \
  --model gpt-5.5 \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --out-root runs_reason_current
```

Equivalent reason command for multiple scenes:

```bash
python test.py \
  --root data \
  --scene-ids 59 242 691 \
  --target-source auto \
  --model gpt-5.5 \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --out-root runs_reason_current
```

## Disable Automatic Reason

To run perception only:

```bash
RUN_REASON_AFTER_PERCEPTION=0 bash run_perception.sh 59
```

## Override Reason Parameters

Examples:

```bash
REASON_MODEL=gpt-4o \
REASON_PRIOR_PROMPT=original \
REASON_RANKING_SCORE=legacy \
bash run_perception.sh 59
```

Run a fixed target id:

```bash
REASON_TARGET_ID=3 bash run_perception.sh 59
```

Use intent resolution:

```bash
REASON_TARGET_SOURCE=intent \
REASON_INSTRUCTION="pick the red cup" \
bash run_perception.sh 59
```

## Reason Outputs

The automatic reason step writes under:

```text
runs_reason_current/<model>/<prior_prompt>/<ranking_score>/
```

Key files:

```text
results.csv
branch_results.json
reason.txt
summary.json
scene_details/scene_<id>.csv
```

`summary.json` and `branch_results.json` include
`selected_graspability_summary`. For a selected object, this includes:

```json
{
  "selected_object_id": 1,
  "selected_object_graspability": 0.8,
  "selected_object_graspability_part_id": 3,
  "selected_object_graspability_parts": {
    "3": 0.9,
    "4": 0.35
  }
}
```

## SAM2 Environment

Perception `vlm` mode still needs SAM2. On didis0 the known working settings
are:

```bash
export SAM2_ROOT=/home/qiuguanhe/miniconda3/envs/smartgrasp/lib/python3.12/site-packages
export SAM2_CHECKPOINT=/home/data/models/torch/hub/checkpoints/sam2.1_hiera_small.pt
```
