"""Safe configuration template. Copy to config.py before using legacy runners.

Keep credentials in environment variables, never in committed source files.
The real local config.py is deliberately excluded from Git.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_CACHE = Path(r"D:\hf_cache\hub\models--Qwen--Qwen3-0.6B")
RULE_VISIBILITY = "provided"
MAX_STEPS = 6
MAX_NEW_TOKENS = 256
SEED = 42

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
