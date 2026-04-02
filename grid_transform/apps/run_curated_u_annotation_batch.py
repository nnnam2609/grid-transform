from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR


CURATED_VTLN_DIR = PROJECT_DIR / "VTLN" / "data"
DEFAULT_MANIFEST_CSV = CURATED_VTLN_DIR / "selection_manifest.csv"
DEFAULT_BATCH_OUTPUT_ROOT = DEFAULT_OUTPUT_DIR / "source_annotation_edits" / "curated_u_batch"
ARTSPEECH_ROOT_DEFAULT = PROJECT_DIR.parent / "Data" / "Artspeech_database"
TARGET_CASE_DEFAULT = "2008-003^01-1791/test"
TARGET_FRAME_DEFAULT = 143020

OUTPUT_BASENAME_RE = re.compile(
    r"^(?P<raw_subject>\d+)_(?P<speaker>P\d+)_(?P<session>S\d+)_F(?P<frame>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BatchCase:
    output_basename: str
    speaker: str
    raw_subject: str
    session: str
    frame_index_1based: int
    annotation_status: str
    annotation_source_path: Path | None
    reference_bundle_dir: Path | None
    reference_bundle_name: str | None


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
        help="Skip cases whose output directory already contains edited_annotation.json.",
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


def parse_manifest_row(row: dict[str, str], vtln_dir: Path) -> BatchCase:
    output_basename = row["output_basename"].strip()
    match = OUTPUT_BASENAME_RE.match(output_basename)
    if match is None:
        raise ValueError(f"Could not parse output_basename={output_basename!r}")
    annotation_source_raw = row.get("annotation_source", "").strip()
    annotation_source_path = None
    if annotation_source_raw.lower().endswith(".zip"):
        annotation_source_path = Path(annotation_source_raw)
    reference_bundle_dir_raw = row.get("reference_bundle_dir", "").strip()
    reference_bundle_dir = Path(reference_bundle_dir_raw) if reference_bundle_dir_raw else None
    reference_bundle_name = row.get("reference_bundle_name", "").strip() or None
    return BatchCase(
        output_basename=output_basename,
        speaker=match.group("speaker").upper(),
        raw_subject=match.group("raw_subject"),
        session=match.group("session").upper(),
        frame_index_1based=int(match.group("frame")),
        annotation_status=row.get("annotation_status", "").strip(),
        annotation_source_path=annotation_source_path,
        reference_bundle_dir=reference_bundle_dir,
        reference_bundle_name=reference_bundle_name,
    )


def load_cases(manifest_csv: Path, vtln_dir: Path) -> list[BatchCase]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_manifest_row(row, vtln_dir) for row in reader]


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


def build_case_output_dir(output_root: Path, case: BatchCase) -> Path:
    return output_root / case.output_basename


def resolve_reference_bundle(case: BatchCase) -> tuple[str, Path, Path, Path]:
    if case.reference_bundle_dir is not None and case.reference_bundle_name is not None:
        reference_speaker = case.reference_bundle_name
        bundle_dir = case.reference_bundle_dir
        zip_path = bundle_dir / f"{reference_speaker}.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"Reference zip not found for override bundle: {zip_path}")
        for ext in (".png", ".tif", ".tiff"):
            image_path = bundle_dir / f"{reference_speaker}{ext}"
            if image_path.is_file():
                return reference_speaker, bundle_dir, image_path, zip_path
        raise FileNotFoundError(
            f"Reference image for override bundle {reference_speaker} was not found in {bundle_dir}."
        )

    if case.annotation_source_path is None:
        raise FileNotFoundError(f"No annotation zip is recorded in the manifest for {case.output_basename}")

    zip_path = case.annotation_source_path
    if not zip_path.is_file():
        raise FileNotFoundError(f"Annotation zip not found: {zip_path}")

    reference_speaker = zip_path.stem
    search_dirs = [zip_path.parent]
    if zip_path.parent.parent.exists() and zip_path.parent.parent not in search_dirs:
        search_dirs.append(zip_path.parent.parent)

    for directory in search_dirs:
        for ext in (".png", ".tif", ".tiff"):
            image_path = directory / f"{reference_speaker}{ext}"
            if image_path.is_file():
                return reference_speaker, zip_path.parent, image_path, zip_path

    raise FileNotFoundError(
        f"Reference image for {reference_speaker} was not found next to {zip_path} or in its parent directory."
    )


def resolve_speaker_root(artspeech_root: Path, speaker: str) -> Path:
    direct = artspeech_root / speaker
    nested = artspeech_root / speaker / speaker
    if (direct / "DCM_2D").exists() and (direct / "OTHER").exists():
        return direct
    if (nested / "DCM_2D").exists() and (nested / "OTHER").exists():
        return nested
    raise FileNotFoundError(
        f"Could not resolve ArtSpeech root for {speaker} under {artspeech_root}. "
        f"Tried {direct} and {nested}."
    )


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
        if args.skip_existing_output and (output_dir / "edited_annotation.json").is_file():
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
