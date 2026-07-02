"""Static VLM configuration for reasoning priors.

Keep API keys in .env. Model and base URL live here so reasoning experiments
do not depend on shell-exported model/provider settings.
"""

VLM_MODEL = "gpt-5.5"
VLM_BASE_URL = "https://www.highland-api.top/v1"
VLM_API_KEY_ENV = "OPENAI_API_KEY"
VLM_TEMPERATURE = 0.0
VLM_TIMEOUT = 300.0
