"""Action catalog and cache.

**Weights live in directory names**, per project convention::

    IDLE/DEFAULT/...     weight 1
    IDLE/NOD-3/...       weight 3

Only a *trailing* ``-<number>`` is read as a weight, so ``WAVE-HAND`` keeps its
name and gets the default weight. The action-name pattern in
:mod:`src.domain.state` already allows ``-``, so nothing else changes.

:class:`ActionCache` mirrors the reference implementation: loading an action
decodes its images into memory (removing the per-frame PNG cost), the oldest
entry is evicted once past ``limit``, the freshly loaded key is protected, and
``eager`` strategy simply raises the limit to "all of them".
"""

from __future__ import annotations

import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.domain.avatar import Avatar
from src.storage import COORDS_FILE

DEFAULT_WEIGHT = 1.0
_WEIGHT_RE = re.compile(r"^(?P<name>.+?)-(?P<weight>\d+(?:\.\d+)?)$")


def parse_action_name(dirname: str) -> tuple[str, float]:
    """``NOD-3`` -> ``("NOD", 3.0)``; ``NOD`` -> ``("NOD", 1.0)``."""
    text = (dirname or "").strip()
    match = _WEIGHT_RE.match(text)
    if not match:
        return text, DEFAULT_WEIGHT
    weight = float(match.group("weight"))
    if weight <= 0:
        return text, DEFAULT_WEIGHT
    return match.group("name"), weight


@dataclass(frozen=True)
class ActionRef:
    """One selectable action: name, weight and the concrete resource directory."""

    behavior: str
    action: str
    weight: float
    dir: Path
    engine: str
    root: Path

    @property
    def key(self) -> str:
        return f"{self.behavior}.{self.action}"


def list_actions(behavior_dir: str | Path, *, engine: str | None = None) -> list[ActionRef]:
    """Every usable action under ``<...>/<BEHAVIOR>/``, with its filename weight."""
    base = Path(behavior_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"behavior directory does not exist: {base}")
    refs: list[ActionRef] = []
    for action_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if engine:
            root = action_dir / engine
            if not (root / COORDS_FILE).is_file():
                continue
            engine_name = engine
        else:
            candidates = sorted(
                child
                for child in action_dir.iterdir()
                if child.is_dir() and (child / COORDS_FILE).is_file()
            )
            if not candidates:
                continue
            root = candidates[0]
            engine_name = root.name
        name, weight = parse_action_name(action_dir.name)
        refs.append(
            ActionRef(
                behavior=base.name,
                action=name,
                weight=weight,
                dir=action_dir,
                engine=engine_name,
                root=root,
            )
        )
    return refs


def find_behavior_dir(avatar_root: str | Path, behavior: str) -> Path:
    """Locate ``<BEHAVIOR>`` under an avatar root (``<root>/<BEHAVIOR>`` or one level down)."""
    root = Path(avatar_root)
    behavior = (behavior or "").strip().upper()
    if not behavior:
        raise ValueError("behavior must not be empty")
    direct = root / behavior
    if direct.is_dir():
        return direct
    nested = sorted(
        child / behavior for child in root.iterdir() if (child / behavior).is_dir()
    )
    if not nested:
        raise FileNotFoundError(f"no {behavior}/ directory under {root}")
    if len(nested) > 1:
        listed = "\n  ".join(str(path) for path in nested)
        raise ValueError(f"{len(nested)} avatars have {behavior}; point --avatar at one:\n  {listed}")
    return nested[0]


def weighted_choice(refs: Sequence[ActionRef], rng: random.Random) -> ActionRef:
    """Pick by weight; all weights zero (or negative) falls back to uniform."""
    weights = [max(0.0, ref.weight) for ref in refs]
    if not any(weights):
        return rng.choice(list(refs))
    return rng.choices(list(refs), weights=weights, k=1)[0]


class ActionCache:
    """LRU of loaded actions; loading decodes the frames into memory."""

    def __init__(self, *, limit: int = 3, strategy: str = "lazy", preload: bool = True):
        if strategy not in {"lazy", "eager"}:
            raise ValueError("action_loading_strategy must be 'lazy' or 'eager'")
        self.strategy = strategy
        self.limit = max(1, int(limit))
        self.preload = bool(preload)
        self._entries: "OrderedDict[Path, Avatar]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def set_total(self, total: int) -> None:
        """Eager strategy keeps every action resident."""
        if self.strategy == "eager":
            self.limit = max(3, int(total))

    def get(self, root: str | Path) -> Avatar:
        root = Path(root)
        cached = self._entries.get(root)
        if cached is not None:
            self._entries.move_to_end(root)
            self.hits += 1
            return cached
        self.misses += 1
        avatar = Avatar.load(root)
        if self.preload:
            avatar.preload()
        self._entries[root] = avatar
        self._trim({root})
        return avatar

    def _trim(self, protected: set[Path]) -> None:
        while len(self._entries) > self.limit:
            root, avatar = next(iter(self._entries.items()))
            if root in protected:
                self._entries.move_to_end(root)
                continue
            self._entries.pop(root)
            avatar.release()

    def stats(self) -> dict:
        return {
            "strategy": self.strategy,
            "limit": self.limit,
            "cached": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "bytes": sum(avatar.loaded_bytes for avatar in self._entries.values()),
        }


__all__ = [
    "ActionCache",
    "ActionRef",
    "DEFAULT_WEIGHT",
    "find_behavior_dir",
    "list_actions",
    "parse_action_name",
    "weighted_choice",
]
