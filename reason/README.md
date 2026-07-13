# SmartGrasp Reason

This directory contains the reasoning stage of SmartGrasp. Perception produces
scene images, object labels, SAM2 parts, and an occlusion matrix. Reason reads
those perception outputs, resolves the target object, classifies the visibility
case, and returns the next grasp action.

The reason stage can be run in two ways:

- Perception plus reason: run `perception/run_perception.sh`; it generates perception
  outputs and then automatically runs one-shot reason on the same scene ids.
- Reason only: run `reason/run_reason.py` directly, or use `reason/run_reason.sh` for comparison
  batches over several ranking modes.

## What Reason Does

For each target object, reason performs these steps:

1. Load `data/scene_<id>/perception/summary.json`.
2. Rebuild the occlusion graph from `occlusion_matrix`.
3. Resolve the target id from `summary.json["annotation"]`, `--target-id`, or
   all visible graph ids.
4. Classify the target into one branch:
   - `fully_visible`
   - `partially_occluded`
   - `fully_occluded`
   - `fault`
5. Run the branch handler and return the current `grasp_id`.
6. Output the selected object, optional selected occluder, candidate scores, and
   object/part graspability.

The default one-shot behavior returns the next object to grasp now. For a
partially occluded target this is usually an occluder to remove first, not the
final target itself. Closed-loop mode can simulate repeatedly removing objects
until the target becomes directly graspable.

## Repository Structure

The main entrypoints are grouped by stage:

- `perception/run_perception.sh`
  Runs perception first and then, by default, runs reason for the same scene ids.
  This is the recommended entrypoint for current perception-to-reason tests.

- `reason/run_reason.py`
  Main reason entrypoint. It finds perception summaries, resolves targets,
  classifies branches, runs the handlers, and writes all reason outputs.

- `reason/run_reason.sh`
  Batch comparison wrapper for reason only. It loops over `MODELS` and
  `ALGORITHMS`, runs `python -m reason.run_reason --closed-loop`, and then runs
  `analyze_reason_experiment.py`.

- `intent/run_intent.py`
  Standalone target-intent resolver. It maps a natural-language instruction or
  `summary.json["annotation"]` to a perception object id.

Inside `reason/`:

- `schemas.py`
  Defines the shared dataclasses: `PerceptionOutput`, `GraspDecision`, and the
  `Branch` enum.

- `data_loader.py`
  Loads `summary.json`, rebuilds the occlusion graph, and attaches optional
  artifacts such as depth, labeled RGB, and the SAM2 part sheet.

- `branch_judge/classifier.py`
  Classifies the target into `fully_visible`, `partially_occluded`,
  `fully_occluded`, or `fault`.

- `fully_visible/handler.py`
  Handles the case where the target is already top-level and directly
  graspable. It returns `grasp_id = target_id` and also scores the target's
  SAM2 parts for graspability.

- `partially_visible/`
  Handles visible targets that are occluded by one or more visible objects.
  - `prior.py`: asks the VLM to score semantic relevance and graspability for
    target ancestors.
  - `geometry.py`: computes graph/geometry priors for top-layer ancestors.
  - `scoring.py`: provides fusion, entropy, and information gain utilities.
  - `handler.py`: selects the best top-layer occluder to remove next.

- `invisible/`
  Handles targets that are not visible in the current graph and are treated as
  fully hidden.
  - `prior.py`: asks the VLM which top-layer visible object may hide the target.
  - `geometry.py`: builds geometry priors and counterfactual candidate sets.
  - `scoring.py`: computes expected information gain after a miss.
  - `handler.py`: selects the top-layer visible object to remove next.

- `graspability.py`
  Shared helper that asks the VLM for object-level and per-SAM2-part
  graspability for the current selected object. This is used so all branches can
  output part graspability.

- `vlm/`
  Shared VLM infrastructure.
  - `config.py`: default model, base URL, API key environment variable, and
    timeout.
  - `client.py`: OpenAI-compatible client and system prompts.
  - `helper.py`: user prompt builders, image encoding, and JSON parsers.

- `intent_handle/`
  Resolves language instructions to perception object ids. This is used when
  `--target-source intent` or the default `--target-source auto` chooses the
  annotation path.

- `closed_loop.py`
  Simulates repeated grasp decisions by removing the selected object from the
  graph and re-running branch logic.

- `THEORY.md` and `THEORY.zh.md`
  Notes for the probability, fusion, entropy, and information gain formulas.

- `inspect_data.py`, `regen_hard_test2.py`, and `temp/`
  Debugging and experiment notes. They are not required for the main pipeline.

## Inputs

Reason reads perception outputs under a scene root, usually:

```text
data/scene_<id>/perception/
```

or for old examples:

```text
sample_data/scene_<id>/perception/
```

The required input is:

```text
data/scene_<id>/perception/summary.json
```

Important fields in `summary.json`:

- `scene_id`: scene id.
- `query_obj_id`: original target id if available.
- `annotation`: natural-language target description.
- `object_points` or `molmo_points`: visible object ids, labels, and points.
- `matrix_labels`: row/column labels for the occlusion matrix.
- `occlusion_matrix`: directed occlusion scores. The convention is
  `row object occludes column object`.
- `object_id_to_sam2_part_ids`: maps object id to SAM2 part ids.
- `object_id_to_sam2_part_files`: maps object id to SAM2 part images.

Useful nearby files:

```text
scene_image.png
depth.npy
label_2_vlm.png
sam2_rgb_parts_sheet.png
sam2_rgb_parts/part_XXX.png
occlusion_graph.json
occlusion_graph.png
```

`label_2_vlm.png` or another labeled image is used by the VLM. The SAM2 part
sheet is used when `--prior-prompt graspability` is enabled.

## Outputs

Reason writes outputs under:

```text
<out-root>/<model>/<prior_prompt>/<ranking_score>/
```

For the default perception-to-reason run this is:

```text
runs_reason_current/gpt-5.5/graspability/ig_graspability/
```

Key files:

- `results.csv`
  One row per tested target. Contains branch, selected `grasp_id`, selected
  object graspability, and status.

- `branch_results.json`
  Full structured result grouped by scene. This includes per-target rows and
  the selected object fields.

- `summary.json`
  Compact summary containing `selected_graspability_summary`. This is the
  easiest JSON file to read downstream when only the selected action matters.

- `reason.txt`
  Human-readable reasoning text. It includes target resolution, branch, selected
  object, and handler messages.

- `scene_details/scene_<id>.csv`
  Candidate-level rows. This is the best place to inspect all candidate scores:
  `P_s`, `P_g`, fused `P`, `IG`, `graspability`, `graspability_parts`, and final
  score.

Important selected-action fields:

```json
{
  "selected_object_id": 3,
  "selected_object_label": "opened Spam can",
  "selected_object_score": 0.42,
  "selected_object_graspability": 0.78,
  "selected_object_graspability_part_id": 3,
  "selected_object_graspability_parts": {
    "3": 0.78
  },
  "selected_object_vlm_result": {
    "P_s": 0.62,
    "P_g": 0.00052,
    "P": 1.0,
    "IG": 0.0,
    "score": 0.0,
    "graspability": 0.78,
    "graspability_part_id": 3,
    "graspability_parts": {
      "3": 0.78
    },
    "vlm_reason": "..."
  }
}
```

`selected_object_*` always describes the current object returned as `grasp_id`:

- In `fully_visible`, it is the target itself.
- In `partially_occluded`, it is the next top-layer occluder to remove.
- In `fully_occluded`, it is the next top-layer visible object to remove.

`selected_occluder_*` is populated only when the selected object is an occluder
or removal candidate, not when the branch is `fully_visible`.

## Branch Logic

Let `G` be the directed occlusion graph. An edge `a -> b` means object `a`
occludes or presses on object `b`.

### fully_visible

The target exists in the graph and has no incoming edges:

```text
target in G
in_degree(target) = 0
```

The target is already top-level, so reason directly returns:

```text
grasp_id = target_id
```

The VLM is still asked to score the target object's SAM2 parts so downstream
code can use part-level graspability.

### partially_occluded

The target exists in the graph but has incoming edges:

```text
target in G
in_degree(target) > 0
```

Reason first finds all ancestors of the target:

```text
A_t = ancestors(target)
```

The VLM can score all ancestors to understand the full occlusion chain. The
final action, however, is chosen only from top-layer ancestors:

```text
C_t = { i in A_t | in_degree(i) = 0 }
```

This means the robot only chooses an object that can be grasped now. If the
chain is `3 -> 4 -> 1`, then for target `1`, object `4` is a direct occluder
but is not top-layer because object `3` is above it. The first one-shot action
is therefore object `3`.

### fully_occluded

The target is not visible in the current graph, but the scene has occlusion
evidence:

```text
target not in G
num_edges(G) > 0
```

Reason treats the target as hidden and considers all top-layer visible objects:

```text
C = { i in visible objects | in_degree(i) = 0 }
```

The VLM estimates which candidate may hide the target, geometry provides a
spatial prior, and expected information gain ranks what to remove next.

### fault

The target is not in the graph and there is no occlusion evidence:

```text
target not in G
num_edges(G) = 0
```

Reason reports a fault because there is no useful basis for occlusion reasoning.

## Scoring Terms

For a candidate object `i`:

- `P_s(i)`: semantic/VLM prior.
- `P_g(i)`: graph/geometry prior.
- `P(i)`: fused belief.
- `IG(i)`: information gain for partial targets.
- `EIG(i)`: expected information gain for fully hidden targets.
- `g(i)`: object-level graspability from the VLM.

The fusion is a product-of-experts normalization:

```text
raw(i) = P_s(i) * P_g(i)
P(i) = raw(i) / sum_j raw(j)
```

Entropy is Shannon entropy in bits:

```text
H(P) = - sum_i P(i) log2 P(i)
```

For partially visible targets, information gain is:

```text
IG(i) = H(P_before) - H(P_after removing i)
```

For fully hidden targets, expected information gain uses the miss-side belief:

```text
EIG(i) = H(P_before) - (1 - P(i)) * H(P_after miss on i)
```

Normalized information gain is:

```text
nIG(i) = max(0, IG(i)) / log2(max(2, |C|))
nEIG(i) = max(0, EIG(i)) / log2(max(2, |C|))
```

## Ranking Algorithms

Select the ranking mode with:

```bash
--ranking-score legacy
--ranking-score ig
--ranking-score ig_graspability
--ranking-score theory
```

The same value can be passed through `perception/run_perception.sh`:

```bash
REASON_RANKING_SCORE=ig_graspability bash perception/run_perception.sh 59
```

### `legacy`

Old behavior.

Partial:

```text
score(i) = P(i) * IG(i)
```

Fully hidden:

```text
score(i) = EIG(i)
```

### `ig`

Ranks only by information gain.

Partial:

```text
score(i) = IG(i)
```

Fully hidden:

```text
score(i) = EIG(i)
```

### `ig_graspability`

Current default for perception-to-reason runs. It keeps structural information
gain but downweights objects that are hard or unsafe to grasp.

Partial:

```text
score(i) = IG(i) * g(i)
```

Fully hidden:

```text
score(i) = EIG(i) * g(i)
```

### `theory`

Normalized theory-style utility.

Partial:

```text
score(i) = g(i) * P(i) * nIG(i)
```

Fully hidden:

```text
score(i) = g(i) * nEIG(i)
```

## Perception Plus Reason

Run from the project root on the remote machine:

```bash
cd /home/admin128/hanhuang/SmartGrasp
conda activate smartgrasp
bash perception/run_perception.sh 59
```

Multiple scenes:

```bash
bash perception/run_perception.sh 59 242 691
```

All scenes configured in the script:

```bash
bash perception/run_perception.sh
```

By default, `perception/run_perception.sh` runs perception and then one-shot reason for the
same scene ids. It does not pass `--closed-loop`.

The reason defaults used by `perception/run_perception.sh` are:

```bash
RUN_REASON_AFTER_PERCEPTION=1
REASON_DATA_ROOT=data
REASON_OUT_ROOT=runs_reason_current
REASON_MODEL=gpt-5.5
REASON_PRIOR_PROMPT=graspability
REASON_RANKING_SCORE=ig_graspability
REASON_TARGET_SOURCE=auto
```

Equivalent direct reason command for one scene:

```bash
python -m reason.run_reason \
  --root data \
  --scene-id 59 \
  --target-source auto \
  --model gpt-5.5 \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --out-root runs_reason_current
```

Disable the automatic reason step and run perception only:

```bash
RUN_REASON_AFTER_PERCEPTION=0 bash perception/run_perception.sh 59
```

Override the reason model or ranking mode:

```bash
REASON_MODEL=gpt-5.5 \
REASON_PRIOR_PROMPT=graspability \
REASON_RANKING_SCORE=theory \
bash perception/run_perception.sh 59
```

Force a fixed target id:

```bash
REASON_TARGET_ID=3 bash perception/run_perception.sh 59
```

Use a custom intent instruction:

```bash
REASON_TARGET_SOURCE=intent \
REASON_INSTRUCTION="the pear under the other pear" \
bash perception/run_perception.sh 242
```

Perception itself searches inputs with this priority:

1. `input/scene_<id>/scene_image.png` or `input/scene_<id>/rgb.png`,
   plus `input/scene_<id>/depth.npy`. If `input.txt` or `instruction.txt`
   exists in the same folder, it overrides the instruction; otherwise
   `summary.json["instruction"]` or `summary.json["annotation"]` is used.
2. `data/scene_<id>/perception/summary.json`,
   `data/scene_<id>/perception/scene_image.png`,
   `data/scene_<id>/perception/depth.npy`
3. fallback source data such as `*.parquet` and `npz_file.zip`. This path does
   not read `input.txt` or `instruction.txt`; it uses the parquet annotation.

Perception `vlm` mode needs SAM2. Set `SAM2_ROOT` and `SAM2_CHECKPOINT` if they
are not discoverable in the environment.

## Reason Only

Reason only assumes perception outputs already exist under `data/` or
`sample_data/`.

### One-shot reason with `reason/run_reason.py`

Run one scene with annotation-based target resolution:

```bash
cd /home/admin128/hanhuang/SmartGrasp
conda activate smartgrasp
python -m reason.run_reason \
  --root data \
  --scene-id 59 \
  --target-source auto \
  --model gpt-5.5 \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --out-root runs_reason_current
```

Run multiple scenes:

```bash
python -m reason.run_reason \
  --root data \
  --scene-ids 59 242 691 \
  --target-source auto \
  --model gpt-5.5 \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --out-root runs_reason_current
```

Run all discovered perception summaries:

```bash
python -m reason.run_reason \
  --root data \
  --target-source auto \
  --model gpt-5.5 \
  --prior-prompt graspability \
  --ranking-score ig_graspability
```

Run all graph objects in one scene instead of using annotation:

```bash
python -m reason.run_reason \
  --root data \
  --scene-id 59 \
  --target-source all \
  --prior-prompt graspability \
  --ranking-score ig_graspability
```

Run one fixed target id:

```bash
python -m reason.run_reason \
  --root data \
  --scene-id 59 \
  --target-source id \
  --target-id 3 \
  --prior-prompt graspability \
  --ranking-score ig_graspability
```

Run intent with a custom instruction:

```bash
python -m reason.run_reason \
  --root data \
  --scene-id 242 \
  --target-source intent \
  --instruction "the pear under the other pear" \
  --prior-prompt graspability \
  --ranking-score ig_graspability
```

Run closed-loop simulation:

```bash
python -m reason.run_reason \
  --root data \
  --scene-id 242 \
  --target-source auto \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --closed-loop \
  --max-steps 20
```

### Reason comparison with `reason/run_reason.sh`

`reason/run_reason.sh` is a bash wrapper for comparison experiments. It runs
`python -m reason.run_reason --closed-loop` for each model and algorithm, then runs
`analyze_reason_experiment.py`.

Example:

```bash
cd /home/admin128/hanhuang/SmartGrasp
conda activate smartgrasp
PYTHON_BIN="$(which python)" \
DATA_ROOT=data \
LIMIT=10 \
OUT_ROOT=runs_reason_compare \
PRIOR_PROMPT=graspability \
MODELS="gpt-5.5" \
ALGORITHMS="legacy ig ig_graspability theory" \
bash reason/run_reason.sh
```

Run only one algorithm:

```bash
PYTHON_BIN="$(which python)" \
DATA_ROOT=data \
LIMIT=1 \
PRIOR_PROMPT=graspability \
ALGORITHMS="ig_graspability" \
bash reason/run_reason.sh
```

`reason/run_reason.sh` environment variables:

- `PYTHON_BIN`: Python executable. Recommended: `$(which python)` after
  `conda activate smartgrasp`.
- `DATA_ROOT`: scene root, usually `data` or `sample_data`.
- `LIMIT`: number of scenes to process from the discovered summaries.
- `OUT_ROOT`: output root, default `runs_reason_compare`.
- `PRIOR_PROMPT`: `original` or `graspability`.
- `MODELS`: space-separated model names.
- `ALGORITHMS`: space-separated ranking modes.
- `USE_PROXY`: set to `1` if proxy variables should be kept.

## `reason/run_reason.py` Arguments

- `--root`
  Root directory containing `scene_<id>/perception/summary.json`. Default:
  `sample_data`.

- `--scene-id`
  Run one scene id.

- `--scene-ids`
  Run a list of scene ids.

- `--target-source`
  How targets are selected. Choices:
  - `auto`: if `--target-id` is set, use it; otherwise if `annotation` exists,
    run intent; otherwise test all graph ids.
  - `all`: test every visible graph id.
  - `id`: use `--target-id`.
  - `intent`: resolve `--instruction` or `summary.json["annotation"]` with the
    VLM intent resolver.

- `--target-id`
  Target id used with `--target-source id`, or used automatically by
  `--target-source auto`.

- `--instruction`
  Natural-language target instruction for `--target-source intent`. If omitted,
  reason uses `summary.json["annotation"]`.

- `--intent-api-key-env`
  Environment variable name for the intent VLM API key. Default:
  `OPENAI_API_KEY`.

- `--intent-base-url`
  OpenAI-compatible base URL for intent resolution.

- `--intent-model`
  VLM model for intent resolution.

- `--intent-timeout`
  Intent request timeout in seconds.

- `--model`
  VLM model for reason priors and graspability. Overrides
  `reason/vlm/config.py`.

- `--out-root`
  Output root. Final output path is
  `<out-root>/<model>/<prior_prompt>/<ranking_score>/`.

- `--csv`
  Override the `results.csv` output path.

- `--json`
  Override the `branch_results.json` output path.

- `--details-dir`
  Override the `scene_details/` output directory.

- `--threshold`
  Minimum occlusion matrix value used when rebuilding graph edges.

- `--prior-prompt`
  Choices:
  - `original`: old VLM prompt. Graspability defaults to compatibility values.
  - `graspability`: asks the VLM for object-level and part-level graspability.

- `--ranking-score`
  Ranking mode. Choices: `legacy`, `ig`, `ig_graspability`, `theory`.

- `--reason-algorithm`
  Compatibility shortcut for older experiments. Choices: `legacy`, `theory`.
  If set, it overrides `--ranking-score`.

- `--closed-loop`
  Simulate repeated removals until the target is graspable or the rollout ends.

- `--max-steps`
  Maximum closed-loop rollout length. Default: `20`.

- `--limit`
  Process only the first `N` discovered scene summaries. Useful for debugging.

## Target Resolution

`--target-source auto` is recommended for current perception-to-reason runs.
Its behavior is:

1. If `--target-id` is provided, use that id.
2. Else if `summary.json["annotation"]` is non-empty, use VLM intent resolution.
3. Else test all graph ids.

The intent resolver uses the current perception scene, object table, labeled
image, and occlusion graph. It returns a perception object id, not necessarily a
FreeGrasp `queryObjId`.

## Environment

Reason uses an OpenAI-compatible VLM client. Set:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://www.highland-api.top/v1
```

`python-dotenv` is optional. If it is installed, `.env` can provide the same
variables. If it is not installed, reason still runs from exported environment
variables.

For current remote usage:

```bash
cd /home/admin128/hanhuang/SmartGrasp
conda activate smartgrasp
```

Use `PYTHON_BIN="$(which python)"` with `reason/run_reason.sh` if the script's default
Python path does not match the current machine.

## Practical Checks

After a run, check:

```text
runs_reason_current/<model>/<prior_prompt>/<ranking_score>/summary.json
runs_reason_current/<model>/<prior_prompt>/<ranking_score>/branch_results.json
runs_reason_current/<model>/<prior_prompt>/<ranking_score>/scene_details/scene_<id>.csv
```

For the selected object's part graspability, read:

```text
selected_graspability_summary[*].selected_object_graspability_parts
```

For all candidate part scores, read:

```text
scene_details/scene_<id>.csv
```

The fields `P_s`, `P_g`, `P`, `IG`, `graspability`, and `score` are included
there whenever the branch produces candidate details.
