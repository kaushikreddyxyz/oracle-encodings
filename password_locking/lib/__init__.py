"""Shared library for the password-locking pipeline.

Loads .env at import time so every entrypoint (samplers included, not just
the trainers' preflight) sees WANDB_API_KEY / HF_TOKEN — needed for gated
weak-base models (Llama-3.2-1B) and authenticated HF download speeds.
"""

from dotenv import load_dotenv

load_dotenv()
