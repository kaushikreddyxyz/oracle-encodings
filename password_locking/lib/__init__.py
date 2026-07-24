"""Shared library for the password-locking pipeline.

Loads .env at import time so every entrypoint (samplers included, not just
the trainers' preflight) sees credentials — needed for gated weak-base
models and authenticated HF download speeds. This repo's .env names the
wandb key WANDB_TOKEN; wandb expects WANDB_API_KEY, so alias it.
"""

import os

from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_TOKEN"):
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
