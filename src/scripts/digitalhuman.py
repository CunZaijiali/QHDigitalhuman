"""Offline digital-human CLI.

    python -m src.scripts.digitalhuman check <path> [--behavior IDLE] [--action DEFAULT] [--json]
    python -m src.scripts.digitalhuman render --action-dir <dir> --audio speech.wav --output out.mp4

Exit codes: 0 ok, 1 usage or resolve error, 2 ``check --strict`` found problems,
3 the requested stage is not wired yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from src.domain.actions import find_behavior_dir
from src.domain.avatar import Avatar, resolve_avatar
from src.domain.models.loaders import describe_models, load_models
from src.engine import create_engine
from src.engine.adapters import create_adapter
from src.engine.asr import load_audio
from src.engine.pipeline import LipSyncInput
from src.session import stream_action
from src.speech import TTSError, synthesize_to_wav
from src.storage import assets_for, resolve_root
from src.utils import find_project_root, get_config, load_environment
from src.utils.ffmpeg import FFMPEG_ENV, encode_with_ffmpeg, find_ffmpeg

NOT_WIRED = 3


def _config() -> dict:
    return get_config()


def _section(config: dict, name: str) -> dict:
    return dict(config.get(name, {}) or {})


def _avatar_root(config: dict) -> Path:
    return resolve_root(_section(config, "paths").get("avatar_root", "avatar"), find_project_root())


def _default_engine(config: dict) -> str:
    return _section(config, "digitalhuman").get("lip_sync_engine") or "musetalkv15"


def _locate(target: Path, config: dict) -> Path:
    """Accept a path from the cwd, or one relative to the configured avatar root."""
    if target.exists():
        return target
    candidate = _avatar_root(config) / target
    return candidate if candidate.exists() else target


# ---------------------------------------------------------------------- check
def cmd_check(args: argparse.Namespace) -> int:
    config = _config()
    try:
        avatar = resolve_avatar(
            _locate(args.target, config),
            behavior=args.behavior,
            action=args.action,
            engine=args.engine,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 1
    payload = avatar.describe()
    payload["problems"] = avatar.problems()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        counts, shapes = payload["counts"], payload["shapes"]
        print(f"action     : {avatar.root}")
        print(f"behavior   : {avatar.behavior}.{avatar.action}  engine={avatar.engine}")
        print(
            f"counts     : full={counts['full_imgs']} face={counts['face_imgs']} "
            f"mask={counts['mask']} coords={counts['coords']} mask_coords={counts['mask_coords']}"
        )
        print(f"shapes     : full={shapes['full']} face={shapes['face']} latents={shapes['latents']}")
        print(f"assets     : {', '.join(payload['required_assets'])}")
        print(f"metadata   : {'present' if payload['metadata'] else 'missing'}")
        if payload["problems"]:
            print("problems   :")
            for problem in payload["problems"]:
                print(f"  - {problem}")
        else:
            print("problems   : none")

    if payload["problems"] and args.strict:
        return 2
    return 0


# --------------------------------------------------------------------- render
def _audio_path_for(args: argparse.Namespace, config: dict) -> tuple[Path | None, int]:
    """Resolve ``--audio`` or ``--text`` into a readable WAV path.

    Returns ``(path, exit_code)``; a ``None`` path means the caller should return
    the given exit code.
    """
    if args.audio:
        path = Path(args.audio)
        if not path.is_file():
            print(f"audio not found: {path}", file=sys.stderr)
            return None, 1
        return path, 0
    if not args.text:
        print("needs --audio <wav> or --text <sentence>", file=sys.stderr)
        return None, 1

    target = Path(tempfile.gettempdir()) / f"qhdh-tts-{uuid4().hex[:8]}.wav"
    try:
        synthesize_to_wav(args.text, target, config=config)
    except TTSError as exc:
        print(f"tts failed: {exc}", file=sys.stderr)
        return None, 1
    print(f"tts        : synthesized {len(args.text)} chars -> {target}", file=sys.stderr)
    return target, 0


def _build_feature_extractor(engine: str, models, config: dict):
    """Build the feature extractor for ``engine``; MuseTalk needs its Whisper."""
    digitalhuman = _section(config, "digitalhuman")
    sample_rate = int(digitalhuman.get("sample_rate", 16000))
    fps = int(digitalhuman.get("fps", 25))
    normalized = engine.strip().lower().replace("_", "")

    if normalized == "wav2lip":
        from src.engine.asr import Wav2LipASR

        return Wav2LipASR(sample_rate=sample_rate, fps=fps)
    if normalized in {"musetalk", "musetalkv15"}:
        if models is None:
            raise ValueError("MuseTalk feature extraction needs the loaded Whisper models")
        from src.engine.asr import MuseTalkASR

        return MuseTalkASR(
            models.whisper, models.feature_extractor, sample_rate=sample_rate, fps=fps
        )
    raise ValueError(f"unsupported engine: {engine}")


def cmd_render(args: argparse.Namespace) -> int:
    config = _config()
    digitalhuman = _section(config, "digitalhuman")
    engine = args.engine or _default_engine(config)
    fps = int(args.fps or digitalhuman.get("fps", 25))
    sample_rate = int(digitalhuman.get("sample_rate", 16000))

    if bool(args.avatar) == bool(args.action_dir):
        print("render needs exactly one of --avatar or --action-dir", file=sys.stderr)
        return 1

    audio_path, code = _audio_path_for(args, config)
    if audio_path is None:
        return code

    try:
        if args.action_dir:
            avatar = Avatar.load(args.action_dir)
        else:
            avatar = resolve_avatar(
                _locate(args.avatar, config),
                behavior=args.behavior,
                action=args.action,
                engine=engine,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 1

    problems = avatar.problems()
    if problems:
        print("action resource has problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    normalized = engine.strip().lower().replace("_", "")
    if avatar.engine and avatar.engine.lower().replace("_", "") != normalized:
        print(
            f"warning: action was built for engine={avatar.engine} but render uses {engine}",
            file=sys.stderr,
        )

    samples = load_audio(audio_path, sample_rate)
    frame_count = max(1, round(samples.size / sample_rate * fps))
    output_path = Path(args.output) if args.output else avatar.root / "render.mp4"
    # Offline rendering has no latency budget, so use a quality-oriented default
    # rather than the low-latency tuning the streaming path needs.
    crf = int(args.crf if args.crf is not None else digitalhuman.get("render_crf", 18))
    preset = str(digitalhuman.get("x264_preset", "") or "") or "medium"

    if args.dry_run:
        plan = {
            "action": str(avatar.root),
            "engine": engine,
            "audio": str(audio_path),
            "samples": int(samples.size),
            "sample_rate": sample_rate,
            "fps": fps,
            "frame_count": frame_count,
            "available_frames": avatar.frame_count,
            "required_assets": list(assets_for(avatar.engine)),
            "blended": bool(avatar.mask_paths and avatar.mask_coords),
            "encoder": "ffmpeg (H.264 + AAC)" if find_ffmpeg() else "cv2 (video only)",
            "crf": crf,
            "output": str(output_path),
        }
        if args.json:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            for key, value in plan.items():
                print(f"{key:17s}: {value}")
        return 0

    try:
        models = load_models(engine, config, find_project_root(), device=args.device)
        adapter = create_adapter(engine, models)
        extractor = _build_feature_extractor(engine, models, config)
        features = extractor.extract(audio_path, frame_count)
        pipeline = create_engine(engine, model=adapter)
        request = LipSyncInput(
            resource_dir=avatar.root,
            audio_features=features,
            output_path=output_path,
            fps=fps,
            sample_rate=sample_rate,
            frame_count=frame_count,
        )
        # Prefer FFmpeg: it yields H.264 + AAC (the clip actually has sound), where
        # cv2.VideoWriter only writes silent MPEG-4 Part 2. Falls back cleanly.
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            try:
                encode_with_ffmpeg(
                    pipeline.iter_frames(request),
                    output_path,
                    audio_path,
                    fps=fps,
                    crf=crf,
                    preset=preset,
                    ffmpeg=ffmpeg,
                )
                frames_written = frame_count
                encoder = "ffmpeg (H.264 + AAC)"
            except RuntimeError as exc:
                print(
                    f"warning: ffmpeg encoding failed, falling back to cv2 "
                    f"(silent MPEG-4): {exc}",
                    file=sys.stderr,
                )
                result = pipeline.process(request)
                frames_written = result.frame_count
                encoder = "cv2 (video only)"
        else:
            print(
                "warning: ffmpeg not found, writing silent MPEG-4 video. Set "
                f"{FFMPEG_ENV} or put ffmpeg on PATH to get H.264 + audio.",
                file=sys.stderr,
            )
            result = pipeline.process(request)
            frames_written = result.frame_count
            encoder = "cv2 (video only)"
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"render failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = {
        "action": str(avatar.root),
        "engine": engine,
        "frames": frames_written,
        "blended": bool(avatar.mask_paths and avatar.mask_coords),
        "encoder": encoder,
        "output": str(output_path),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key:10s}: {value}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """Load the configured models for real; the fastest way to prove the wiring."""
    config = _config()
    engine = args.engine or _default_engine(config)
    try:
        models = load_models(engine, config, find_project_root(), device=args.device)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"models failed: {exc}", file=sys.stderr)
        return 1
    payload = describe_models(engine, models)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key:10s}: {value}")
    return 0


# ---------------------------------------------------------------------- stream
def cmd_stream(args: argparse.Namespace) -> int:
    """Render one action and push it to an RTMP server in real time."""
    config = _config()
    digitalhuman = _section(config, "digitalhuman")
    engine = args.engine or _default_engine(config)
    # fps is the content timeline (audio features and action frames are indexed by
    # it), not merely a delivery rate -- lowering it to "keep up" silently slows
    # the lips and the motion down. Falling behind is handled by skipping frames.
    fps = int(args.fps or digitalhuman.get("fps", 25))
    sample_rate = int(digitalhuman.get("sample_rate", 16000))
    url = args.url  # resolved below, once we know whether an rtmp sink was asked for

    if bool(args.avatar) == bool(args.action_dir):
        print("stream needs exactly one of --avatar or --action-dir", file=sys.stderr)
        return 1
    sink_names = [part.strip().lower() for part in (args.sink or "rtmp").split(",") if part.strip()]
    if not sink_names:
        print("--sink needs at least one value", file=sys.stderr)
        return 1
    # Only fall back to the configured push_url when an rtmp sink is actually wanted,
    # otherwise a camera-only run would claim to be pushing to rtmp.
    if url is None and any(name in {"rtmp", "push"} for name in sink_names):
        url = digitalhuman.get("push_url")
    config = dict(config)
    if args.bitrate:
        config["digitalhuman"] = {**dict(config.get("digitalhuman", {}) or {}),
                                  "bitrate": int(args.bitrate)}
    session_overrides = dict(config.get("session", {}) or {})
    if args.batch:
        session_overrides["render_batch_size"] = max(1, int(args.batch))
    if args.camera_backend:
        session_overrides["virtual_camera_backend"] = args.camera_backend
    if session_overrides:
        config["session"] = session_overrides
    if any(name in {"rtmp", "push"} for name in sink_names) and not url:
        print(
            "the rtmp sink needs --url (or [digitalhuman].push_url); "
            "use --sink virtualcam for a camera-only run",
            file=sys.stderr,
        )
        return 1

    audio_path, code = _audio_path_for(args, config)
    if audio_path is None:
        return code

    try:
        if args.action_dir:
            avatar, behavior_dir = Avatar.load(args.action_dir), None
        elif args.behavior:
            # Rotate across every action of this behaviour, weighted by directory name.
            behavior_dir = find_behavior_dir(_locate(args.avatar, config), args.behavior)
            avatar = None
        else:
            avatar, behavior_dir = (
                resolve_avatar(
                    _locate(args.avatar, config),
                    behavior=args.behavior,
                    action=args.action,
                    engine=engine,
                ),
                None,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"stream failed: {exc}", file=sys.stderr)
        return 1

    if avatar is not None:
        problems = avatar.problems()
        if problems:
            print("action resource has problems:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

    try:
        result = stream_action(
            audio_path,
            url,
            avatar=avatar,
            behavior_dir=behavior_dir,
            engine=engine,
            sink_names=sink_names,
            fps=fps,
            sample_rate=sample_rate,
            duration=args.duration,
            device=args.device,
            config=config,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"stream failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = {
        "url": result.url,
        "action": result.action,
        "engine": engine,
        "frames": result.frames,
        "seconds": round(result.seconds, 2),
        "first_frame_seconds": result.first_frame_seconds,
        "render_fps": round(result.render_fps, 2),
        "realtime_ratio": round(result.realtime_ratio, 3),
        "stats": result.stats,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"target     : {result.url}")
        print(f"action     : {result.action}")
        print(
            f"rendered   : {result.frames} frames at {result.render_fps:.1f} fps "
            f"({result.realtime_ratio * 100:.0f}% of the {fps} fps target)"
        )
        print(f"latency    : first frame on the wire after {result.first_frame_seconds:.2f}s")
        print(
            f"streamed   : video={result.stats.get('video_streamed')} "
            f"(repeated {result.stats.get('video_repeated')}, dropped "
            f"{result.stats.get('video_dropped')}) audio={result.stats.get('audio_streamed')} packets"
        )
        print(f"total      : {result.seconds:.1f}s wall clock")
        rotations = result.stats.get("rotations") or []
        if rotations:
            print(f"rotation   : {len(rotations)} switch(es) -> {', '.join(rotations)}")
        cache = result.stats.get("cache") or {}
        if cache:
            print(
                f"cache      : {cache.get('cached')}/{cache.get('limit')} resident "
                f"({cache.get('strategy')}), hits={cache.get('hits')} misses={cache.get('misses')}, "
                f"{cache.get('bytes', 0) / 1024**2:.0f} MB"
            )
        by_sink = result.stats.get("by_sink") or {}
        for name, info in by_sink.items():
            if name == "rtmp":
                continue
            if name == "speaker":
                print(f"audio      : playing={info.get('playing')} error={info.get('error')}")
                continue
            print(
                f"camera     : {info.get('device')!r} ({info.get('backend')}) "
                f"frames={info.get('video_streamed')} error={info.get('error')}"
            )
        print(
            f"pacing     : batch={result.stats.get('render_batch')} frames, "
            f"skipped={result.stats.get('skipped_frames')} to stay on the audio clock, "
            f"content lag={result.stats.get('sync_lag_seconds_avg')}s avg / "
            f"{result.stats.get('sync_lag_seconds_max')}s max"
        )
        if failed := result.stats.get("failed_sinks"):
            print(f"WARNING    : dropped failing sink(s): {', '.join(failed)}")
        print(f"worker     : alive={result.stats.get('worker_alive')} error={result.stats.get('error')}")
        if "rtmp" in (result.stats.get("sinks") or [result.stats.get("sink") or "rtmp"]):
            print(f"watch      : {_hls_hint(result.url)}")
    return 0


def _hls_hint(url: str) -> str:
    """Best-effort HLS URL for the MediaMTX setup, else the raw RTMP URL."""
    marker = "://"
    if marker not in url:
        return url
    host_path = url.split(marker, 1)[1]
    host, _, path = host_path.partition("/")
    hostname = host.split(":")[0]
    return f"http://{hostname}:8888/{path}/index.m3u8"


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digitalhuman", description="Offline digital-human CLI."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser("check", help="validate one action resource")
    check.add_argument("target", type=Path, help="action dir, avatar root, or a path under avatar_root")
    check.add_argument("--behavior")
    check.add_argument("--action")
    check.add_argument("--engine")
    check.add_argument("--json", action="store_true")
    check.add_argument("--strict", action="store_true", help="exit 2 when problems are found")
    check.set_defaults(func=cmd_check)

    render = subcommands.add_parser("render", help="offline audio -> video (skeleton)")
    render.add_argument("--avatar", type=Path, help="avatar root or a path under avatar_root")
    render.add_argument("--action-dir", type=Path, help="the action directory itself")
    render.add_argument("--behavior")
    render.add_argument("--action")
    render.add_argument("--engine", help="default: [digitalhuman].lip_sync_engine")
    render.add_argument("--audio", type=Path)
    render.add_argument("--text", help="sentence to synthesize with TTS (instead of --audio)")
    render.add_argument(
        "--crf",
        type=int,
        help="H.264 quality for the mp4 (0-51, lower is better; default [digitalhuman].render_crf)",
    )
    render.add_argument("--output", type=Path)
    render.add_argument("--fps", type=int)
    render.add_argument("--device", help="override [digitalhuman].device")
    render.add_argument("--dry-run", action="store_true", help="print the plan without loading models")
    render.add_argument("--json", action="store_true")
    render.set_defaults(func=cmd_render)

    models = subcommands.add_parser(
        "models", help="load the configured models and report what came back"
    )
    models.add_argument("--engine", help="default: [digitalhuman].lip_sync_engine")
    models.add_argument("--device", help="override [digitalhuman].device")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    stream = subcommands.add_parser("stream", help="render one action and push it over RTMP")
    stream.add_argument("--avatar", type=Path, help="avatar root or a path under avatar_root")
    stream.add_argument("--action-dir", type=Path, help="the action directory itself")
    stream.add_argument("--behavior")
    stream.add_argument("--action")
    stream.add_argument("--engine", help="default: [digitalhuman].lip_sync_engine")
    stream.add_argument("--audio", type=Path, help="a WAV to speak")
    stream.add_argument("--text", help="sentence to synthesize with TTS (instead of --audio)")
    stream.add_argument(
        "--sink",
        default="rtmp",
        help="comma separated destinations: rtmp, virtualcam, speaker (default: rtmp)",
    )
    stream.add_argument(
        "--bitrate",
        type=int,
        help="RTMP video bitrate in bps (default [digitalhuman].bitrate)",
    )
    stream.add_argument(
        "--batch",
        type=int,
        help="frames per inference batch (default [session].render_batch_size)",
    )
    stream.add_argument(
        "--camera-backend",
        choices=["obs", "unitycapture"],
        help="override [session].virtual_camera_backend",
    )
    stream.add_argument("--url", help="default: [digitalhuman].push_url")
    stream.add_argument("--fps", type=int)
    stream.add_argument("--duration", type=float, help="limit the pushed audio to N seconds")
    stream.add_argument("--device", help="override [digitalhuman].device")
    stream.add_argument("--json", action="store_true")
    stream.set_defaults(func=cmd_stream)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
