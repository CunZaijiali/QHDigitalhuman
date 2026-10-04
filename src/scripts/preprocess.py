"""CLI: build one avatar action asset from a video or a still image.

    python -m src.scripts.preprocess source.mp4 --behavior IDLE --action DEFAULT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessing import create_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess one avatar action asset.")
    parser.add_argument("source", type=Path, help="source video or image")
    parser.add_argument("--avatar-root", default=None, help="override [paths].avatar_root")
    parser.add_argument("--index", type=int, default=None, help="avatar index (default: next free)")
    parser.add_argument("--avatar-id", default=None, help="avatar uuid (default: random uuid4)")
    parser.add_argument("--behavior", default="IDLE")
    parser.add_argument("--action", default="DEFAULT")
    parser.add_argument("--engine", default=None, help="default: [digitalhuman].lip_sync_engine")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", action="store_true", help="force still-image input")
    group.add_argument("--video", action="store_true", help="force video input")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    is_image = True if args.image else (False if args.video else None)
    result = create_action(
        args.source,
        avatar_root=args.avatar_root,
        index=args.index,
        avatar_id=args.avatar_id,
        behavior=args.behavior,
        action=args.action,
        engine=args.engine,
        is_image=is_image,
    )
    print(
        json.dumps(
            {
                "root": str(result.root),
                "frames": result.frame_count,
                "faces": result.face_count,
                "missed": result.missed_frames,
                "output_shape": result.output_shape,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
