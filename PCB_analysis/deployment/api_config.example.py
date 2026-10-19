"""Copy to api_config.py and fill in the official contest Qwen endpoint/key.

``api_config.py`` is intentionally ignored by Git.  The local pipeline works
without this file while both API switches remain disabled.
"""

API_ENABLED = False
API_ENDPOINT = "https://<workspace>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = ""
API_MODEL = "qwen3.7-plus"

API_TIMEOUT_SECONDS = 30
API_MAX_RETRIES = 1
API_MAX_REQUESTS_PER_IMAGE = 30
API_SEND_IMAGE = True
API_IMAGE_MAX_SIDE = 1280
API_TEMPERATURE = 0.0
API_MAX_TOKENS = 300
API_ENABLE_THINKING = False
API_JSON_MODE = True

# API is semantic/OCR disambiguation only.  It must never replace local box,
# pin-tip, wire or edge geometry.
API_USE_FOR_TEXT_ONLY = True
