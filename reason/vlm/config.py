"""VLM configuration for reasoning priors.

Keep provider credentials in ``.env``.  The model default remains stable for
the experiment while the endpoint follows ``OPENAI_BASE_URL`` when configured.
"""

import os

VLM_MODEL = "gpt-5.5"
VLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
VLM_API_KEY_ENV = "OPENAI_API_KEY"
VLM_TEMPERATURE = 0.0
VLM_TIMEOUT = 600.0
VLM_MAX_RETRIES = 0
