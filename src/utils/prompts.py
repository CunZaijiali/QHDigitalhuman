from pathlib import Path

from src.utils import find_project_root


PROMPTS_ROOT = find_project_root() / "prompts"


def load_prompt(name: str, **values: object) -> str:
    """Read a prompt at call time and replace ``{{PLACEHOLDER}}`` values."""
    path = PROMPTS_ROOT / name
    template = path.read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("{{" + key.upper() + "}}", str(value))
    return template.strip()
