r"""Batch export warped + review videos for all completed workspaces.

By default skips workspaces whose videos already exist (use --no-skip-existing to re-export).

Usage:
  .\.venv\Scripts\python .\scripts\run\run_batch_export_workspace_videos.py
  .\.venv\Scripts\python .\scripts\run\run_batch_export_workspace_videos.py --max-frames 100
  .\.venv\Scripts\python .\scripts\run\run_batch_export_workspace_videos.py --output-mode review
  .\.venv\Scripts\python .\scripts\run\run_batch_export_workspace_videos.py --no-skip-existing
  .\.venv\Scripts\python .\scripts\run\run_batch_export_workspace_videos.py --limit 5 --max-frames 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Bootstrap: add project root to sys.path
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from grid_transform.annotation_to_grid_workflow import step_file_paths
from grid_transform.apps.export_saved_workspace_video_threaded import run_workspace_export
from grid_transform.config import DEFAULT_OUTPUT_DIR


DEFAULT_WORKSPACE_ROOT = DEFAULT_OUTPUT_DIR / "annotation_to_grid_transform"

_BAR_WIDTH = 40


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_progress(done: int, total: int, elapsed: float, succeeded: int, failed: int, skipped: int) -> None:
    frac = done / total if total else 0.0
    filled = int(_BAR_WIDTH * frac)
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    pct = frac * 100.0

    if done > 0 and elapsed > 0:
        eta_sec = elapsed / done * (total - done)
        eta_str = _format_duration(eta_sec)
        elapsed_str = _format_duration(elapsed)
        speed = f"  elapsed={elapsed_str}  ETA={eta_str}"
    else:
        speed = ""

    line = f"\r[{bar}] {done}/{total} ({pct:.0f}%)  ok={succeeded} err={failed} skip={skipped}{speed}   "
    sys.stdout.write(line)
    sys.stdout.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export videos for all completed annotation-to-grid workspaces.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help="Root folder containing workspace subdirectories.",
    )
    parser.add_argument("--workers", type=int, default=6, help="Threaded workers per workspace.")
    parser.add_argument("--prefetch", type=int, default=0, help="Prefetch count (0 = auto).")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per workspace (0 = full session).")
    parser.add_argument(
        "--output-mode",
        choices=("both", "warped", "review"),
        default="both",
        help="Which videos to produce.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N workspaces (0 = all).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip workspaces whose output video(s) already exist (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-export even if output video(s) already exist.",
    )
    return parser.parse_args(argv)


def find_completed_workspaces(root: Path) -> list[Path]:
    """Return workspace dirs that have a saved transform_spec.latest.json."""
    results = []
    if not root.is_dir():
        return results
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        paths = step_file_paths(child)
        if paths["transform_spec"].is_file() and paths["selection"].is_file():
            results.append(child)
    return results


def _video_exists(workspace_dir: Path, output_mode: str) -> bool:
    """Return True if the expected output video(s) for this workspace already exist."""
    export_dir = workspace_dir / "background_export"
    if not export_dir.is_dir():
        return False
    if output_mode in ("both", "warped"):
        if not list(export_dir.glob("*_warped.mp4")):
            return False
    if output_mode in ("both", "review"):
        if not list(export_dir.glob("*_review.mp4")):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspaces = find_completed_workspaces(args.workspace_root)
    if not workspaces:
        print(f"No completed workspaces found in {args.workspace_root}")
        return 0

    if args.limit > 0:
        workspaces = workspaces[: args.limit]

    total = len(workspaces)
    skip_label = "on" if args.skip_existing else "off"
    print(
        f"Found {total} completed workspace(s).  "
        f"mode={args.output_mode}  workers={args.workers}  "
        f"max-frames={'full' if args.max_frames == 0 else args.max_frames}  "
        f"skip-existing={skip_label}"
    )
    print()

    succeeded = 0
    failed = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    start_time = time.monotonic()
    done = 0

    for workspace_dir in workspaces:
        name = workspace_dir.name

        if args.skip_existing and _video_exists(workspace_dir, args.output_mode):
            skipped += 1
            done += 1
            _print_progress(done, total, time.monotonic() - start_time, succeeded, failed, skipped)
            continue

        t0 = time.monotonic()
        try:
            payload = run_workspace_export(
                workspace_dir=workspace_dir,
                workers=args.workers,
                prefetch=args.prefetch,
                max_frames=args.max_frames,
                output_mode=args.output_mode,
            )
            elapsed_ws = time.monotonic() - t0
            mode = payload.get("mode", "?")
            if mode == "preview_only":
                detail = f"preview only ({payload.get('reason', '?')})"
            else:
                parts = []
                if payload.get("warped_video"):
                    parts.append(Path(payload["warped_video"]).name)
                if payload.get("review_video"):
                    parts.append(Path(payload["review_video"]).name)
                fc = payload.get("frame_count", "?")
                detail = f"{fc} frames  {' | '.join(parts)}  ({elapsed_ws:.0f}s)"
            succeeded += 1
            result_line = f"  OK   {name}\n       {detail}"
        except Exception as exc:
            elapsed_ws = time.monotonic() - t0
            errors.append((name, str(exc)))
            failed += 1
            result_line = f"  ERR  {name}\n       {exc}"

        done += 1
        # Clear progress line, print result, then redraw progress
        sys.stdout.write("\r" + " " * 120 + "\r")
        print(result_line)
        _print_progress(done, total, time.monotonic() - start_time, succeeded, failed, skipped)

    # Final newline after progress bar
    sys.stdout.write("\n\n")
    total_elapsed = time.monotonic() - start_time
    print(f"Done.  succeeded={succeeded}  failed={failed}  skipped={skipped}  total={total}  elapsed={_format_duration(total_elapsed)}")

    if errors:
        print("\nFailed workspaces:")
        for ws_name, msg in errors:
            print(f"  {ws_name}: {msg}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
