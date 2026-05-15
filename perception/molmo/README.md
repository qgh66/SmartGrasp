# molmo-annotator

A small reusable module that runs **allenai/Molmo-7B-D-0924** to extract object points from an image, then produces:

- `molmo_label.png` (numbered labels drawn on the image, matplotlib style)
- `molmo_points.json` (pixel coordinates)

## Model weights (NOT included)

Weights are downloaded automatically via Hugging Face Transformers the first time you run.

Default cache directory is typically:
- `~/.cache/huggingface/hub`

You can override the download/cache location:

```bash
export HF_HOME=/abs/path/to/hf_cache
# or:
export HF_HUB_CACHE=/abs/path/to/hf_cache/hub
```

## Install

From a zip:
```bash
pip install ./molmo_annotator.zip
```

From source folder:
```bash
pip install .
```

## CLI usage

```bash
molmo-annotate \
  --image image.png \
  --prompt "Point out the objects in the red rectangle on the table." \
  --out out
```

Outputs:
- `out/molmo_label.png`
- `out/molmo_points.json`

## Python usage

```python
from molmo_annotator import MolmoAnnotator

ann = MolmoAnnotator(model_id="allenai/Molmo-7B-D-0924")
res = ann.annotate_to_folder("image.png", "Point out all objects in the green tray", "out")

print(res["base64_png"][:80])
print(res["json_path"])
print(res["points"][:3])
```

## Sample command

```bash
molmo-annotate   --image 1.jpg   --prompt 'Point out all graspable objects on the table. Use short noun labels.'   --out ./out
```