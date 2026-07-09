# Reason Module

This package turns perception outputs into grasp decisions.

## Overview

The reasoning pipeline takes a `PerceptionOutput` and decides what the robot
should grasp next.

Core steps:

1. Load a scene summary and rebuild the occlusion graph.
2. Classify the target into one branch:
   - `fully_visible`
   - `partially_occluded`
   - `fully_occluded`
   - `fault`
3. Run the matching branch handler.
4. Optionally simulate a closed loop by removing the chosen object and repeating.

The reason module consumes perception outputs. It does not generate masks or
occlusion graphs by itself. The expected input directory is usually:

```text
sample_data/scene_<id>/perception/
```

or a live perception result:

```text
data/scene_<id>/perception/
```

The key input file is:

```text
summary.json
```

The loader also tries to attach nearby `depth.npy`, `label_3_final.png`, and
`mask/*.png` files when they exist.

## Main Files

- `schemas.py`
  Defines the shared input/output dataclasses used by all branches.

- `data_loader.py`
  Loads `summary.json`, rebuilds the graph, and attaches optional artifacts such
  as depth, labeled RGB, and per-object masks.

- `branch_judge/classifier.py`
  Decides which branch should handle the current target.

- `closed_loop.py`
  Runs a branch handler, removes the selected object from the graph, and repeats
  until the target becomes directly graspable or the rollout stops.

## Branches

### `fully_visible/`

Used when the target is already on top of the occlusion graph.

- `handler.py`
  Returns a terminal decision that directly grasps the target.

Rule: the target id exists in the graph and has no incoming occlusion edges.

### `partially_visible/`

Used when the target is visible in the graph but still has occluders above it.

Files:

- `prior.py`
  Calls a VLM to score which visible occluders are semantically relevant.

- `geometry.py`
  Scores candidates by graph paths from the candidate to the target.

- `scoring.py`
  Fuses semantic and geometric priors, estimates structural gain after removal,
  computes a simple cost, and combines them into a ranking score.

- `handler.py`
  Selects a top-level visible occluder to remove next.

Rule: the target id exists in the graph, but one or more objects occlude it.
The handler ranks visible blockers by semantic prior, geometric prior,
information gain, and removal cost.

### `invisible/`

Used when the target is missing from the graph and is assumed to be fully hidden.

Files:

- `prior.py`
  Calls a VLM to estimate which visible object is most likely hiding the target.

- `geometry.py`
  Builds a geometry cache and computes an area/height proxy for each visible
  candidate. It also supports counterfactual candidate updates after a removal.

- `scoring.py`
  Computes expected information gain by rebuilding the belief after a
  counterfactual miss.

- `handler.py`
  Chooses the top-level visible object with the highest expected gain.

Rule: the target id is not in the graph, but the scene still has occlusion
edges, so the target is treated as hidden behind visible objects.

### `fault`

Used when the target id is missing from the graph and the scene has no
occlusion evidence. This usually means the target is not represented in the
current perception result.

## VLM Support

`vlm/` contains the shared VLM client used by semantic prior modules.

- `client.py`
  Defines the abstract client, the OpenAI implementation, and the system prompts.

- `helper.py`
  Builds user prompts, encodes labeled RGB images, and parses VLM scores.

## Intent Handling

`intent_handle/` resolves a natural-language instruction to a perception object
id before the branch-specific reasoning code runs.

The intent input is the current perception scene, not the original FreeGrasp
object ids:

- `summary.json`: object table, `molmo_points`, `annotation`, and occlusion
  matrix.
- `label_3_final.png`: labeled RGB image whose visible ids match the perception
  object ids.
- `occlusion_graph.json` / `occlusion_graph.png`: occlusion structure used to
  choose the least-occluded candidate when multiple objects match.

The output is a predicted perception id. This id matches `summary.json`
`molmo_id` / the label id in `label_3_final.png`; it is not guaranteed to match
FreeGrasp `queryObjId` or `groundTruthObjIds`.

Run intent on all sample perception scenes:

```bash
/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python run_intent.py --use sample
```

For each scene, this reads:

```text
sample_data/scene_<id>/perception/summary.json
sample_data/scene_<id>/perception/label_3_final.png
sample_data/scene_<id>/perception/occlusion_graph.json
sample_data/scene_<id>/perception/occlusion_graph.png
```

It uses `summary.json["annotation"]` as the instruction and writes:

```text
sample_data/scene_<id>/intent_id/id.txt
```

`id.txt` contains the predicted perception object id, or `none` if no valid
target is selected.

Run only one sample scene:

```bash
/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python run_intent.py --use sample --scene-id 1094
```

This intent step is useful before branch reasoning when the target is described
only by language. It maps:

```text
annotation + labeled RGB + object table + occlusion graph
```

to:

```text
target perception id
```

That predicted id can then be used as the target id for branch classification
and closed-loop reasoning.

## Batch Reasoning Evaluation

`test.py` runs branch classification and handler dispatch over perception
summaries.

Run all sample perception scenes:

```bash
/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python test.py --root sample_data --model gpt-4o
```

Run closed-loop reasoning:

```bash
/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python test.py --root sample_data --model gpt-4o --closed-loop
```

Run one scene:

```bash
/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python test.py --root sample_data --scene-id 1094 --model gpt-4o --closed-loop
```

Run one target id inside one scene:

```bash
/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python test.py --root sample_data --scene-id 1094 --target-id 11 --model gpt-4o --closed-loop
```

Outputs are written under:

```text
runs_detail/<model>/
```

Important files:

- `results.csv`: one row per tested target.
- `branch_results.json`: structured per-scene results.
- `scene_details/scene_<id>.csv`: candidate-level scores such as `P_s`, `P_g`,
  fused belief `P`, information gain `IG`, cost, and final score.

## Score Semantics

For the probability and entropy derivations behind these fields, see
[`THEORY.md`](THEORY.md). 中文版本见 [`THEORY.zh.md`](THEORY.zh.md)。

For partially visible targets, the score is computed over visible occluder
candidates:

- `P_s`: semantic prior from the VLM.
- `P_g`: geometric prior from graph structure.
- `P`: normalized fusion of `P_s * P_g`.
- `IG`: information gain estimated after counterfactually removing a candidate.
- `cost`: approximate difficulty of removing that candidate.
- `score`: final ranking score, using belief-weighted information gain minus a
  small cost penalty.

For invisible targets, the belief is over top-level visible candidates that may
hide the target. The information gain estimates how much uncertainty is reduced
if the selected candidate is removed and the target is still not found.

## Typical Usage

Load one scene and classify a target:

```python
from reason.data_loader import load_sample
from reason.branch_judge.classifier import classify_branch

perception = load_sample("sample_data/scene_365/gt/summary.json")
branch, reason = classify_branch(perception)
print(branch, reason)
```

Run the closed loop:

```python
from reason.closed_loop import run_closed_loop

result = run_closed_loop(perception, max_steps=20)
print(result.success, result.final_status)
```

## Notes

- `partially_visible` is the most exercised branch in the current sample set.
- `fully_occluded` / `invisible` code exists, but it needs explicit test cases to
  validate end-to-end behavior.
- The closed-loop simulator currently does not execute the `fully_occluded`
  branch; it stops with a status message instead.

## intent_handler:
