from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_transform.curated_batch import (
    ARTSPEECH_ROOT_DEFAULT,
    CURATED_VTLN_DIR,
    DEFAULT_BATCH_OUTPUT_ROOT,
    DEFAULT_MANIFEST_CSV,
    TARGET_CASE_DEFAULT,
    TARGET_FRAME_DEFAULT,
    BatchCase,
    build_case_output_dir,
    load_cases,
    resolve_reference_bundle,
    resolve_speaker_root,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the existing cv2 source-annotation editor sequentially for the canonical "
            "ArtSpeech /u/ VTLN/data bundle. Each case opens in order; press 'v' in the editor "
            "to save the edited annotation and render the grid/video outputs."
        )
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=CURATED_VTLN_DIR,
        help="Folder containing the curated PNG/ZIP reference bundle.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=DEFAULT_MANIFEST_CSV,
        help="CSV manifest created for the curated /u/ bundle.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_BATCH_OUTPUT_ROOT,
        help="Root output folder. Each case writes into its own subdirectory.",
    )
    parser.add_argument(
        "--artspeech-root",
        type=Path,
        default=ARTSPEECH_ROOT_DEFAULT,
        help="Root of Artspeech_database. The launcher resolves direct or nested speaker folders from here.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional subset filter. Accepts speaker ids (for example P7), raw subject ids, or output basenames.",
    )
    parser.add_argument(
        "--start-at",
        help="Optional resume point. Accepts the same identifiers as --only and starts from the first match.",
    )
    parser.add_argument(
        "--skip-existing-output",
        action="store_true",
        help="Skip cases whose output directory already contains save_summary.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later cases if one case fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without launching the cv2 editor.",
    )
    parser.add_argument(
        "--target-frame",
        type=int,
        default=TARGET_FRAME_DEFAULT,
        help="nnUNet target frame number passed through to edit_source_annotation.py.",
    )
    parser.add_argument(
        "--target-case",
        default=TARGET_CASE_DEFAULT,
        help="nnUNet target case passed through to edit_source_annotation.py.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("both", "warped", "review"),
        default="both",
        help="Sequence output mode passed through to edit_source_annotation.py.",
    )
    parser.add_argument(
        "--max-output-frames",
        type=int,
        default=0,
        help="Optional debug frame cap passed through to edit_source_annotation.py.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=8,
        help="Number of threaded workers used when you save+render from the editor. Default: 8.",
    )
    parser.add_argument(
        "--render-prefetch",
        type=int,
        default=0,
        help="Optional number of in-flight frames for the threaded renderer. Use 0 for the default heuristic.",
    )
    parser.add_argument(
        "--window-mode",
        choices=("full-height", "fullscreen", "fixed"),
        default="full-height",
        help="Initial cv2 window mode passed through to edit_source_annotation.py.",
    )
    parser.add_argument(
        "--skip-video-on-save",
        action="store_true",
        help="Pass through to the editor. Useful only with --no-gui.",
    )
    parser.add_argument(
        "--render-background",
        action="store_true",
        help="Launch the sequence render as a detached background job after saving the annotation.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Pass through to the editor. Saves/renders headlessly instead of opening cv2.",
    )
    return parser.parse_args(argv)
def normalized_tokens(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    return {value.strip().casefold() for value in values if value.strip()}


def case_matches(case: BatchCase, token: str) -> bool:
    probe = token.casefold()
    return probe in {
        case.output_basename.casefold(),
        case.speaker.casefold(),
        case.raw_subject.casefold(),
    }


def filter_cases(cases: list[BatchCase], only_tokens: list[str] | None) -> list[BatchCase]:
    probes = normalized_tokens(only_tokens)
    if not probes:
        return cases
    return [case for case in cases if any(case_matches(case, token) for token in probes)]


def apply_start_at(cases: list[BatchCase], start_at: str | None) -> list[BatchCase]:
    if not start_at:
        return cases
    for index, case in enumerate(cases):
        if case_matches(case, start_at):
            return cases[index:]
    raise ValueError(f"--start-at value {start_at!r} did not match any planned case")
def build_run_plan(args: argparse.Namespace, cases: list[BatchCase]) -> tuple[list[BatchCase], list[dict[str, object]]]:
    runnable: list[BatchCase] = []
    skipped: list[dict[str, object]] = []
    for case in cases:
        reasons: list[str] = []
        try:
            resolve_reference_bundle(case)
        except FileNotFoundError:
            reasons.append("missing_reference_bundle")
        try:
            resolve_speaker_root(args.artspeech_root, case.speaker)
        except FileNotFoundError:
            reasons.append("missing_dataset_root")
        output_dir = build_case_output_dir(args.output_root, case)
        if args.skip_existing_output and (output_dir / "save_summary.json").is_file():
            reasons.append("existing_output")
        if reasons:
            skipped.append(
                {
                    "output_basename": case.output_basename,
                    "speaker": case.speaker,
                    "session": case.session,
                    "frame": case.frame_index_1based,
                    "annotation_status": case.annotation_status,
                    "reasons": reasons,
                }
            )
            continue
        runnable.append(case)
    return runnable, skipped


def plan_summary(
    args: argparse.Namespace,
    selected_cases: list[BatchCase],
    runnable_cases: list[BatchCase],
    skipped_cases: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "vtln_dir": str(args.vtln_dir),
        "manifest_csv": str(args.manifest_csv),
        "output_root": str(args.output_root),
        "artspeech_root": str(args.artspeech_root),
        "render_workers": int(args.render_workers),
        "render_prefetch": int(args.render_prefetch),
        "render_background": bool(args.render_background),
        "selected_cases": len(selected_cases),
        "runnable_cases": len(runnable_cases),
        "skipped_cases": skipped_cases,
        "planned_order": [
            {
                "output_basename": case.output_basename,
                "speaker": case.speaker,
                "session": case.session,
                "frame": case.frame_index_1based,
                "dataset_root": str(resolve_speaker_root(args.artspeech_root, case.speaker)),
                "reference_speaker": resolve_reference_bundle(case)[0],
                "reference_image": str(resolve_reference_bundle(case)[2]),
                "reference_zip": str(resolve_reference_bundle(case)[3]),
                "output_dir": str(build_case_output_dir(args.output_root, case)),
            }
            for case in runnable_cases
        ],
    }


def build_editor_argv(args: argparse.Namespace, case: BatchCase) -> list[str]:
    dataset_root = resolve_speaker_root(args.artspeech_root, case.speaker)
    reference_speaker, reference_vtln_dir, _reference_image, _reference_zip = resolve_reference_bundle(case)
    argv = [
        "--artspeech-speaker",
        case.speaker,
        "--session",
        case.session,
        "--reference-speaker",
        reference_speaker,
        "--source-frame",
        str(case.frame_index_1based),
        "--target-frame",
        str(args.target_frame),
        "--target-case",
        args.target_case,
        "--dataset-root",
        str(dataset_root),
        "--vtln-dir",
        str(reference_vtln_dir),
        "--output-dir",
        str(build_case_output_dir(args.output_root, case)),
        "--output-mode",
        args.output_mode,
        "--max-output-frames",
        str(args.max_output_frames),
        "--render-workers",
        str(args.render_workers),
        "--render-prefetch",
        str(args.render_prefetch),
        "--window-mode",
        args.window_mode,
    ]
    if args.skip_video_on_save:
        argv.append("--skip-video-on-save")
    if args.render_background:
        argv.append("--render-background")
    if args.no_gui:
        argv.append("--no-gui")
    return argv


def launch_case(args: argparse.Namespace, case: BatchCase, index: int, total: int) -> dict[str, object]:
    from grid_transform.apps import edit_source_annotation

    output_dir = build_case_output_dir(args.output_root, case)
    summary_path = output_dir / "save_summary.json"
    reference_speaker, reference_vtln_dir, reference_image, reference_zip = resolve_reference_bundle(case)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{index}/{total}] Launching {case.output_basename} "
        f"-> {case.speaker}/{case.session} frame {case.frame_index_1based}"
    )
    print(f"    reference bundle: {reference_image.name} + {reference_zip.name}")
    print(f"    reference speaker: {reference_speaker}")
    print(f"    reference vtln dir: {reference_vtln_dir}")
    print(f"    dataset root: {resolve_speaker_root(args.artspeech_root, case.speaker)}")
    print(f"    output dir: {output_dir}")
    print("    editor keys: s = save only, v = save + render, q = next case, x = exit all")
    editor_args = edit_source_annotation.parse_args(build_editor_argv(args, case))
    if not editor_args.no_gui and edit_source_annotation.cv2 is None:
        raise SystemExit(
            "cv2 is not installed in this environment. Install opencv-python in the repo venv first, "
            "or use --no-gui for headless save/render only."
        )
    editor = edit_source_annotation.SourceAnnotationEditor(editor_args)
    editor_result = editor.run() or {}
    editor_action = str(editor_result.get("action", "next"))
    if summary_path.is_file():
        status = "saved"
    elif editor_action == "exit_all":
        status = "exit_all_without_save"
    else:
        status = "exited_without_save"
    return {
        "output_basename": case.output_basename,
        "speaker": case.speaker,
        "session": case.session,
        "frame": case.frame_index_1based,
        "output_dir": str(output_dir),
        "save_summary_json": str(summary_path) if summary_path.is_file() else "",
        "editor_action": editor_action,
        "status": status,
        "exit_code": 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.manifest_csv, args.vtln_dir)
    selected_cases = apply_start_at(filter_cases(cases, args.only), args.start_at)
    runnable_cases, skipped_cases = build_run_plan(args, selected_cases)
    summary = plan_summary(args, selected_cases, runnable_cases, skipped_cases)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0

    results: list[dict[str, object]] = []
    for index, case in enumerate(runnable_cases, start=1):
        try:
            result = launch_case(args, case, index, len(runnable_cases))
            results.append(result)
            if result.get("editor_action") == "exit_all":
                print(json.dumps({"batch_status": "stopped_by_user", "after_case": case.output_basename}, indent=2, ensure_ascii=False))
                break
        except SystemExit as exc:
            if exc.code not in (None, 0):
                error_payload = {
                    "output_basename": case.output_basename,
                    "status": "failed",
                    "error": f"SystemExit({exc.code})",
                }
                results.append(error_payload)
                print(json.dumps(error_payload, indent=2, ensure_ascii=False))
                if not args.continue_on_error:
                    raise
        except Exception as exc:
            error_payload = {
                "output_basename": case.output_basename,
                "status": "failed",
                "error": str(exc),
            }
            results.append(error_payload)
            print(json.dumps(error_payload, indent=2, ensure_ascii=False))
            if not args.continue_on_error:
                raise

    print(json.dumps({"completed_runs": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
