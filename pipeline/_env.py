"""Environment bootstrap shared by the stages that make live calls.

Secrets stay in the environment and are never echoed. A shell variable wins over `.env`, which
CI relies on, so a conflict is reported rather than silently resolved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        return
    env_path = REPO_ROOT / ".env"
    for key, value in dotenv_values(env_path).items():
        current = os.environ.get(key)
        if value and current and current.strip() != value.strip():
            print(
                f"warning: {key} is set in your shell and differs from .env; "
                "the shell value is being used",
                file=sys.stderr,
            )
    load_dotenv(env_path, override=False)
