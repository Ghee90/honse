"""Environment and .env loader so team keys never live in source code."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(env_path: str | None = None) -> None:
    """Loads key=value pairs from .env into os.environ without overriding existing env vars."""
    candidates = [env_path] if env_path else [
        os.path.join(HERE, ".env"),
        os.path.join(os.path.dirname(HERE), "HONSE-Junction", ".env"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            break
    if "SPREAD_URL" not in os.environ:
        os.environ["SPREAD_URL"] = "https://spread.bahnexpress.com"
