# GraspNet Dash GUI

Browser-based viewer for saved GraspNet simulation results.

## Run

```bash
conda activate smartgrasp
cd /home/admin128/beilei/graspnet-workspace
python gui/app.py \
  --host 0.0.0.0 \
  --port 8050 \
  --results results_phase3_002/results.json \
  --viz-data results_phase3_002/results_viz_data.pkl
```

Open:

```text
http://127.0.0.1:8050
```

## Inputs

- `results.json`: grasp scores, poses, widths, depths, lift result, success labels.
- `*_viz_data.pkl`: RGB image, depth image, point cloud, and optional grasp trajectories.

The GUI also scans the workspace for compatible result JSON files and lets you
switch between them from the sidebar.
