from __future__ import annotations

import json
import os
import re
import sys
from html import unescape
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from .draw import draw_labeled_image_matplotlib
from .io import ensure_dir, image_file_to_base64_png, write_json
from .schema import points_to_jsonable


# --- simple global cache to avoid re-loading the 7B model multiple times in one process ---
_GLOBAL: Dict[str, Any] = {"processor": None, "model": None, "model_id": None}


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _log(message: str) -> None:
    print(f"[molmo-annotator] {message}", file=sys.stderr, flush=True)


class MolmoAnnotator:
    """
    Reusable Molmo annotator:
    - infer pixel points with ids (and optional open-vocabulary label)
    - draw labeled png (matplotlib style)
    - write points JSON
    - optionally return base64 for GPT image_url usage
    """

    def __init__(
        self,
        model_id: str = "allenai/Molmo-7B-D-0924",
        device_map: str | None = None,   # keep for API compatibility, but we do NOT use device_map="auto"
        torch_dtype: str = "auto",
        device: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.last_generated_text = ""
        self.last_parse_mode = ""

    def _load(self) -> None:
        if _GLOBAL["model"] is not None and _GLOBAL["processor"] is not None and _GLOBAL["model_id"] == self.model_id:
            return

        device = self.device or ("cuda" if _has_cuda() else "cpu")
        if self.torch_dtype == "auto":
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
        elif self.torch_dtype in {"float16", "fp16"}:
            dtype = torch.float16
        elif self.torch_dtype in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif self.torch_dtype in {"float32", "fp32"}:
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported torch_dtype={self.torch_dtype!r}")
        _log(f"loading processor for {self.model_id}")

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        # Important:
        # - do NOT pass device_map="auto" (it can lead to mixed CPU/GPU tensors for this model)
        # - low_cpu_mem_usage=True reduces CPU RAM peak during load (helps avoid OOM-kill)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        _log(f"moving model to {device}")
        model.to(device)
        model.eval()

        _GLOBAL["processor"] = processor
        _GLOBAL["model"] = model
        _GLOBAL["model_id"] = self.model_id

    @property
    def processor(self):
        self._load()
        return _GLOBAL["processor"]

    @property
    def model(self):
        self._load()
        return _GLOBAL["model"]

    # ----------------------------
    # Output-format prompting
    # ----------------------------
    @staticmethod
    def build_structured_prompt(user_prompt: str) -> str:
        """
        Ask Molmo to return native point tags with labels.
        x/y are in [0,100] percentage coordinates; we convert them to pixels.
        """
        return (
            user_prompt.strip()
            + "\n\n"
            + "Point to the center of each distinct physical object that matches the request. "
              "Do not mark the same object more than once. "
              "Return one object per line using Molmo point tags in this shape:\n"
              "<point x=\"actual_x\" y=\"actual_y\" alt=\"short_label\">short_label</point>\n"
              "Use x and y as 0-100 image percentage coordinates. "
              "Use alt and the tag text as a short noun category/name. "
              "Replace actual_x and actual_y with the real coordinates from this image. "
              "Output ONLY these point tags, no extra text.\n"
        )

    # ----------------------------
    # Parsers
    # ----------------------------
    @staticmethod
    def _coord_to_pixel(value: float, image_size: int) -> int:
        if 0.0 <= value <= 100.0:
            return int((value / 100.0) * image_size)
        return int(value)

    @staticmethod
    def _valid_point(x: float, y: float, image_w: int, image_h: int) -> bool:
        percent_coords = 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0
        pixel_coords = 0.0 <= x <= float(image_w) and 0.0 <= y <= float(image_h)
        return percent_coords or pixel_coords

    @staticmethod
    def _clean_label(label: str) -> str:
        label = re.sub(r"<[^>]+>", " ", unescape(str(label)))
        label = re.sub(r"\s+", " ", label).strip(" \t\r\n:;,.\"'")
        return label

    @staticmethod
    def _attrs_from_text(text: str) -> Dict[str, str]:
        attrs: Dict[str, str] = {}
        pattern = r"([A-Za-z_][\w:-]*)\s*=\s*(['\"])(.*?)\2"
        for key, _, value in re.findall(pattern, text, flags=re.DOTALL):
            attrs[key.lower()] = unescape(value).strip()
        return attrs

    @staticmethod
    def _dedupe_points(
        points: List[Tuple[int, int, int, str]],
        image_w: int,
        image_h: int,
    ) -> List[Tuple[int, int, int, str]]:
        """
        Remove repeated marks from model output and renumber the remaining points.
        Dedupe ignores model-generated ids because repetition usually changes only id.
        """
        deduped: List[Tuple[int, int, int, str]] = []
        same_label_thresh = max(18, int(min(image_w, image_h) * 0.03))
        any_label_thresh = max(8, int(min(image_w, image_h) * 0.015))

        for _, x, y, label in points:
            norm_label = MolmoAnnotator._clean_label(label).lower()
            duplicate = False

            for _, kept_x, kept_y, kept_label in deduped:
                kept_norm_label = MolmoAnnotator._clean_label(kept_label).lower()
                dist_sq = (x - kept_x) ** 2 + (y - kept_y) ** 2
                if norm_label and norm_label == kept_norm_label:
                    if dist_sq <= same_label_thresh ** 2:
                        duplicate = True
                        break
                elif dist_sq <= any_label_thresh ** 2:
                    duplicate = True
                    break

            if not duplicate:
                deduped.append((len(deduped) + 1, x, y, label))

        return deduped

    @classmethod
    def _json_obj_to_point(
        cls,
        obj: Any,
        image_w: int,
        image_h: int,
        fallback_id: int,
    ) -> Tuple[int, int, int, str] | None:
        if not isinstance(obj, dict):
            return None

        x_keys = ("x", "point_x", "cx", "center_x")
        y_keys = ("y", "point_y", "cy", "center_y")
        x_key = next((k for k in x_keys if k in obj), None)
        y_key = next((k for k in y_keys if k in obj), None)
        if x_key is None or y_key is None:
            return None

        try:
            molmo_id = int(obj.get("id", obj.get("molmo_id", obj.get("object_id", fallback_id))))
            x = float(obj[x_key])
            y = float(obj[y_key])
        except Exception:
            return None

        if not cls._valid_point(x, y, image_w, image_h):
            return None

        label = ""
        for key in ("label", "category", "name", "object", "class"):
            if key in obj:
                label = cls._clean_label(str(obj[key]))
                break

        pixel_x = cls._coord_to_pixel(x, image_w)
        pixel_y = cls._coord_to_pixel(y, image_h)
        return molmo_id, pixel_x, pixel_y, label

    @staticmethod
    def extract_points(molmo_output: str, image_w: int, image_h: int) -> List[Tuple[int, int]]:
        """
        Fallback parser: Extract (pixel_x, pixel_y) from Molmo generated text.
        Molmo is assumed to output x/y in [0, 100] percentage coordinates using x=".." y="..".
        """
        points: List[Tuple[int, int]] = []
        pattern = r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"'
        for match in re.finditer(pattern, molmo_output):
            x, y = float(match.group(1)), float(match.group(2))
            if MolmoAnnotator._valid_point(x, y, image_w, image_h):
                pixel_x = MolmoAnnotator._coord_to_pixel(x, image_w)
                pixel_y = MolmoAnnotator._coord_to_pixel(y, image_h)
                points.append((pixel_x, pixel_y))
        return points

    @staticmethod
    def extract_objects_jsonl(molmo_output: str, image_w: int, image_h: int) -> List[Tuple[int, int, int, str]]:
        """
        Primary parser: Parse JSONL objects from model output:
          {"id":1,"x":..,"y":..,"label":"..."}
        Return [(molmo_id, pixel_x, pixel_y, label), ...]
        """
        results: List[Tuple[int, int, int, str]] = []
        seen_points = set()

        def add_loaded(value: Any) -> None:
            items: List[Any]
            if isinstance(value, list):
                items = value
            elif isinstance(value, dict) and isinstance(value.get("objects"), list):
                items = value["objects"]
            else:
                items = [value]

            for item in items:
                point = MolmoAnnotator._json_obj_to_point(item, image_w, image_h, len(results) + 1)
                if point is not None:
                    if point in seen_points:
                        continue
                    seen_points.add(point)
                    results.append(point)

        candidates = [molmo_output]
        candidates.extend(ln.strip() for ln in molmo_output.splitlines() if ln.strip())
        candidates.extend(match.group(0) for match in re.finditer(r"\{[^{}]*\}", molmo_output, flags=re.DOTALL))
        candidates.extend(match.group(0) for match in re.finditer(r"\[[\s\S]*?\]", molmo_output))

        seen = set()
        for candidate in candidates:
            text = candidate.strip()
            if text in seen:
                continue
            seen.add(text)
            if text.startswith("```"):
                text = re.sub(r"^```(?:json|jsonl)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
                continue
            try:
                add_loaded(json.loads(text))
            except Exception:
                continue

        if not results:
            return results

        # Prefer stable ordering by id
        try:
            return sorted(results, key=lambda t: t[0])
        except Exception:
            return results

    @staticmethod
    def extract_objects_point_markup(molmo_output: str, image_w: int, image_h: int) -> List[Tuple[int, int, int, str]]:
        """
        Parse Molmo-style point markup such as:
          <point x="52.1" y="33.4" alt="cup">cup</point>
          <point x1="52.1" y1="33.4" x2="18.0" y2="40.0" alt="cup">...</point>
        Return [(molmo_id, pixel_x, pixel_y, label), ...].
        """
        results: List[Tuple[int, int, int, str]] = []

        tag_pattern = r"<point\b(?P<attrs>[^>]*)>(?P<body>.*?)</point>|<point\b(?P<self_attrs>[^>]*)/?>"
        for match in re.finditer(tag_pattern, molmo_output, flags=re.IGNORECASE | re.DOTALL):
            attrs_text = match.group("attrs") or match.group("self_attrs") or ""
            body = match.group("body") or ""
            attrs = MolmoAnnotator._attrs_from_text(attrs_text)

            suffixes = []
            for key in attrs:
                suffix_match = re.fullmatch(r"x(\d*)", key)
                if suffix_match and f"y{suffix_match.group(1)}" in attrs:
                    suffixes.append(suffix_match.group(1))
            suffixes = sorted(set(suffixes), key=lambda s: int(s) if s else 0)

            for suffix in suffixes:
                try:
                    x = float(attrs[f"x{suffix}"])
                    y = float(attrs[f"y{suffix}"])
                except Exception:
                    continue
                if not MolmoAnnotator._valid_point(x, y, image_w, image_h):
                    continue

                label = ""
                for key in (f"label{suffix}", f"alt{suffix}", "label", "alt", "name", "category"):
                    if key in attrs:
                        label = MolmoAnnotator._clean_label(attrs[key])
                        break
                if not label:
                    label = MolmoAnnotator._clean_label(body)

                results.append((
                    len(results) + 1,
                    MolmoAnnotator._coord_to_pixel(x, image_w),
                    MolmoAnnotator._coord_to_pixel(y, image_h),
                    label,
                ))

        if results:
            return results

        # Last chance: parse generic x/y/alt attributes even if the model omitted <point>.
        pattern = (
            r'x(?P<i>\d*)="\s*(?P<x>[0-9]+(?:\.[0-9]+)?)"\s+'
            r'y(?P=i)="\s*(?P<y>[0-9]+(?:\.[0-9]+)?)"'
            r'(?:[^<>\n]*?(?:alt|label)(?P=i)?="\s*(?P<label>[^"]*?)")?'
        )
        for match in re.finditer(pattern, molmo_output, flags=re.IGNORECASE):
            x = float(match.group("x"))
            y = float(match.group("y"))
            if not MolmoAnnotator._valid_point(x, y, image_w, image_h):
                continue
            results.append((
                len(results) + 1,
                MolmoAnnotator._coord_to_pixel(x, image_w),
                MolmoAnnotator._coord_to_pixel(y, image_h),
                MolmoAnnotator._clean_label(match.group("label") or ""),
            ))

        return results

    # ----------------------------
    # Inference
    # ----------------------------
    def infer_points(self, image: Image.Image, prompt: str) -> List[Tuple[int, int, int, str]]:
        """
        Return [(molmo_id, pixel_x, pixel_y, label), ...]
        label is open-vocabulary; may be empty if fallback parsing is used.
        """
        image_w, image_h = image.size

        # Use structured prompt to get labels
        structured_prompt = self.build_structured_prompt(prompt)

        _log("preparing inputs")
        inputs = self.processor.process(images=[image], text=structured_prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # For strict structured output, deterministic decode is usually more reliable
        gen_cfg = GenerationConfig(
            max_new_tokens=250,
            do_sample=False,
            stop_strings=["<|endoftext|>"],
        )

        _log("generating")
        if _has_cuda():
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                output = self.model.generate_from_batch(inputs, gen_cfg, tokenizer=self.processor.tokenizer)
        else:
            output = self.model.generate_from_batch(inputs, gen_cfg, tokenizer=self.processor.tokenizer)

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.last_generated_text = generated_text
        _log("parsing output")

        # 1) Try JSONL objects with labels
        objs = self.extract_objects_jsonl(generated_text, image_w, image_h)
        if objs:
            self.last_parse_mode = "json"
            return self._dedupe_points(objs, image_w, image_h)

        # 2) Try Molmo native <point ... alt="label"> markup
        objs = self.extract_objects_point_markup(generated_text, image_w, image_h)
        if objs:
            self.last_parse_mode = "point_markup"
            return self._dedupe_points(objs, image_w, image_h)

        # 3) Fallback to old x=".." y=".." format (no labels)
        pts = self.extract_points(generated_text, image_w, image_h)
        self.last_parse_mode = "xy_fallback" if pts else "none"
        return self._dedupe_points([(i + 1, x, y, "") for i, (x, y) in enumerate(pts)], image_w, image_h)

    # ----------------------------
    # End-to-end
    # ----------------------------
    def annotate_to_folder(
        self,
        image_path: str,
        prompt: str,
        out_dir: str,
        labeled_png_name: str = "molmo_label.png",
        json_name: str = "molmo_points.json",
        return_base64: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end:
        - load image
        - infer points (+ optional label)
        - draw labeled image (matplotlib, consistent with original)
        - write JSON (pixel coords + label)
        - optionally return base64 of labeled png

        Returns dict with:
          - points: [(molmo_id,x,y,label), ...]
          - labeled_png_path
          - json_path
          - base64_png (optional)
          - json_data (the dict written)
        """
        ensure_dir(out_dir)

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        points = self.infer_points(image, prompt)

        # draw only uses (id, x, y); ignore label if present
        points_for_draw = [(p[0], p[1], p[2]) for p in points]

        labeled_png_path = os.path.join(out_dir, labeled_png_name)
        draw_labeled_image_matplotlib(image=image, points_with_ids=points_for_draw, out_png_path=labeled_png_path)

        json_path = os.path.join(out_dir, json_name)
        json_data: Dict[str, Any] = {
            "model_id": self.model_id,
            "prompt": prompt,
            "image": {"path": image_path, "width": int(w), "height": int(h)},
            "parse_mode": self.last_parse_mode,
            "raw_model_output": self.last_generated_text,
            "points": points_to_jsonable(points),
        }
        write_json(json_path, json_data)

        res: Dict[str, Any] = {
            "points": points,
            "labeled_png_path": labeled_png_path,
            "json_path": json_path,
            "json_data": json_data,
        }

        if return_base64:
            res["base64_png"] = image_file_to_base64_png(labeled_png_path)

        return res
