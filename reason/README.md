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

## VLM Support

`vlm/` contains the shared VLM client used by semantic prior modules.

- `client.py`
  Defines the abstract client, the OpenAI implementation, and the system prompts.

- `helper.py`
  Builds user prompts, encodes labeled RGB images, and parses VLM scores.

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
