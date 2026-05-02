import os
from pathlib import Path


PLACEHOLDER_ENV_VALUES = {
    "your_api_key_here",
    "replace_with_a_long_random_secret",
}


def load_dotenv_if_available(*args, **kwargs):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(*args, **kwargs)


def read_env_file(env_path):
    env_path = Path(env_path)
    if not env_path.exists():
        return {}
    values = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def is_configured_env_value(value):
    if value is None:
        return False
    return bool(str(value).strip()) and not is_placeholder_env_value(value)


def is_placeholder_env_value(value):
    if value is None:
        return False
    return str(value).strip().lower() in PLACEHOLDER_ENV_VALUES


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
