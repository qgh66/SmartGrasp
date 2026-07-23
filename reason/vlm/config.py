"""Static VLM configuration for reasoning priors.

API key and base URL are read from the project-root api_config.json.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
with open(_ROOT / "api_config.json", encoding="utf-8") as _f:
    _cfg = json.load(_f)

VLM_MODEL = "gpt-5.5"
VLM_BASE_URL = _cfg["base_url"]
VLM_API_KEY_ENV = "OPENAI_API_KEY"
VLM_TEMPERATURE = 0.0
VLM_TIMEOUT = 600.0
VLM_MAX_RETRIES = 0
