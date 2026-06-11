#!/usr/bin/env python3
"""Quick test to verify the VLM (OpenAI-compatible) API connectivity."""

import os
import sys
import time

# --- Config (mirrors run_perception.sh + sam2_langsam_pipeline.py) ---
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-0icgaaSMWa6ZBEmzKE960dC35DPmPuzUzN7hTGuFofOUCcHm")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://www.highland-api.top/v1")
MODEL_ID = os.environ.get("REVIEW_MODEL_ID", "gpt-5.5")
TIMEOUT = 30.0
# -----------------------------------------------------------------

def test_text_chat() -> bool:
    """Test basic text chat connectivity."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

    print(f"[1/3] Testing text chat to {BASE_URL} with model={MODEL_ID} ...")
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip()
        print(f"  ✓ Response: {text}")
        return True
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        return False


def test_vision_api() -> bool:
    """Test vision API — mirrors the project's actual usage pattern."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

    print(f"[2/3] Testing vision API (list objects in a 1x1 pixel dummy image) ...")
    try:
        # Minimal 1x1 red pixel PNG in base64
        dummy_image_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        response = client.responses.create(
            model=MODEL_ID,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image in one word."},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{dummy_image_b64}"},
                    ],
                }
            ],
            max_output_tokens=20,
            store=False,
        )
        text = getattr(response, "output_text", str(response))
        print(f"  ✓ Response: {text[:120]}")
        return True
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        return False


def test_scene_objects_prompt() -> bool:
    """Test the actual scene-objects prompt used in the pipeline."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

    print(f"[3/3] Testing scene-objects listing prompt (text-only, no image) ...")
    try:
        response = client.responses.create(
            model=MODEL_ID,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "This is a test. Pretend you see a scene with: "
                                "a red apple on the left, a blue cup in the center, a silver spoon on the right. "
                                "Return ONLY JSON with this schema: "
                                '{"objects":[{"id":1,"description":"...","relative_position":"...","visible_parts":["..."]}]}.'
                            ),
                        },
                    ],
                }
            ],
            max_output_tokens=200,
            store=False,
        )
        text = getattr(response, "output_text", str(response))
        print(f"  ✓ Response: {text[:200]}")
        return True
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        return False


def main() -> int:
    print("=" * 64)
    print("SmartGrasp VLM API Connectivity Test")
    print(f"  BASE_URL = {BASE_URL}")
    print(f"  MODEL_ID = {MODEL_ID}")
    print(f"  API_KEY  = {API_KEY[:12]}...{API_KEY[-4:]}")
    print("=" * 64 + "\n")

    t0 = time.time()
    results = {
        "text_chat": test_text_chat(),
        "vision_api": test_vision_api(),
        "scene_objects_prompt": test_scene_objects_prompt(),
    }
    elapsed = time.time() - t0

    print(f"\n{'=' * 64}")
    print(f"Results ({elapsed:.1f}s):")
    all_ok = True
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False
    print(f"{'=' * 64}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
