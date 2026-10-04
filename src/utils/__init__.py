import tomli as tomllib
from pathlib import Path
from typing import Any


def find_project_root(marker: Path | str = "pyproject.toml") -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Can`t find {marker} in project root.")


def get_config(config_path: Path | str = "config.toml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = find_project_root() / path
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Can`t find {config_path}.")
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc


def load_environment(dotenv_path: Path | str | None = None) -> bool:
    """Load ``.env`` from the project root into the process environment.

    Idempotent and non-destructive: existing environment variables win, because
    ``override=False``. Returns True when a file was actually read. Call this from
    entry points, not from library code.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional for library-only use
        return False
    path = Path(dotenv_path) if dotenv_path else find_project_root() / ".env"
    return bool(load_dotenv(path, override=False))


__all__ = [
    "find_project_root",
    "get_config",
    "load_environment",
]
