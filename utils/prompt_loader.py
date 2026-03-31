"""
Loader for YAML-based LLM prompt configuration files in the prompts/ directory.
"""
from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_cache: dict[str, dict] = {}


def load_prompt(name: str) -> dict:
    """Load and cache a prompt config from prompts/{name}.yaml.

    Args:
        name: Prompt name without extension (e.g. 'sql_generation').

    Returns:
        dict with at minimum: model (str), max_tokens (int), user_template (str).
        May also contain: system (str), retry_suffix (str).

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    if name not in _cache:
        path = _PROMPTS_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with path.open() as f:
            _cache[name] = yaml.safe_load(f)
    return _cache[name]
