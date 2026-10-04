"""QHDigitalHuman -- 轻量数字人资产管理与渲染推流服务。

本项目的引擎代码放在仓库的 ``src/`` 树下（``python -m src.scripts.*``），
并不随 wheel 一起安装。这个包只提供一个 ``qhdigitalhuman`` 命令入口，
面向"克隆仓库后使用"的场景，因此会先把仓库根加入 ``sys.path``。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the digitalhuman CLI (``check`` / ``models`` / ``render`` / ``stream``)."""
    root = Path(__file__).resolve().parents[2]
    if (root / "src" / "scripts" / "digitalhuman.py").is_file() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.scripts.digitalhuman import main as cli
    except ImportError as exc:  # pragma: no cover - only when installed as a wheel
        raise SystemExit(
            "qhdigitalhuman 需要在仓库根目录运行（引擎代码在 src/ 下，不随包安装）。\n"
            f"原始导入错误: {exc}"
        ) from exc
    return cli(argv)


__all__ = ["main"]
