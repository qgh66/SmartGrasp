# SmartGrasp FreeGrasp RSR evaluation

This directory is self-contained. It does not modify `run_pipeline.sh`,
`perception/`, `intent/`, or `reason/`.

## Dataset layout

`prepare_inputs.py` reads:

- `data/npz_file.zip`
- `data/train-00000-of-00002.parquet`
- `data/train-00001-of-00002.parquet`

It deterministically selects 20 globally unique scenes from each of the six
`(difficulty, ambiguity)` categories. Only cases with splits `0, 1, 2` and
three distinct annotation strings are eligible.

```bash
$HOME/anaconda3/envs/smartgrasp/bin/python -m rsr_hh.prepare_inputs
```

The resulting layout is entirely under `rsr_hh/data/`:

```text
rsr_hh/data/input/
  manifest.json
  01_hard_ambiguous/
    scene_59/
      metadata.json
      scene_image.png
      depth.npy
      instances_objects.npy
      source.npz
      annotations/
        split_0/{instruction.txt,metadata.json}
        split_1/{instruction.txt,metadata.json}
        split_2/{instruction.txt,metadata.json}
```

## Execution

Run the full fixed matrix. Perception runs once per scene and is reused by all
three annotations and all four methods. The current comparison fixes the model
to GPT-5.5 when invoked with ``--reason-model gpt-5.5``:

- `information_gain_original`: original prompt + `ig`
- `information_gain_graspability`: graspability prompt + `ig_graspability`
- `theory_original`: original prompt + `theory` (graspability defaults to 1)
- `theory_graspability`: graspability prompt + `theory`

```bash
bash rsr_hh/run_rsr_hh.sh
```

Useful smoke-test commands:

```bash
# Required smoke test: two scenes, GPT-4o, both algorithms, three annotations.
bash rsr_hh/run_rsr_hh.sh --testcase hard_ambiguous --limit-scenes 2 \
  --reason-model gpt-4o --fail-fast

# Perception only.
bash rsr_hh/run_rsr_hh.sh --testcase hard_ambiguous --limit-scenes 1 --perception-only

# Reuse existing perception and run one annotation only.
bash rsr_hh/run_rsr_hh.sh --testcase hard_ambiguous --limit-scenes 1 --reason-only --split 0
```

Outputs are isolated per annotation while sharing one perception directory:

```text
rsr_hh/data/output/
  perception/01_hard_ambiguous/scene_59/perception/  # generated once
  results/
    gpt-5.5/information_gain_original/01_hard_ambiguous/scene_59/annotations/...
    gpt-5.5/information_gain_graspability/...
    gpt-5.5/theory_original/...
    gpt-5.5/theory_graspability/...
```

Each annotation-specific `perception/` is a lightweight view whose artifacts
are symbolic links to the shared perception result. Intent and reason outputs
are real, separate files.

## RSR

SmartGrasp perception IDs are mapped to the FreeGrasp ground-truth instance ID
by maximum mask overlap, with point lookup as a fallback. Parquet IDs are
zero-based; NPZ labels are one-based because label zero is background.

Missing or failed runs count as failures and remain in the RSR denominator,
following the current ``rsr_hh/evaluate.py`` behavior. Compute or recompute the
metric with:

```bash
$HOME/anaconda3/envs/smartgrasp/bin/python -m rsr_hh.evaluate
```

Final files:

- `rsr_hh/data/output/reports/<model>/<algorithm>/rsr_results.csv`
- `rsr_hh/data/output/reports/<model>/<algorithm>/rsr_summary.json`
- `rsr_hh/data/output/reports/rsr_matrix_summary.json`
- `rsr_hh/data/output/run_failures.json`
