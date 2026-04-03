from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.artspeech_video import load_session_data, normalize_frame
from grid_transform.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VTLN_DIR,
    TONGUE_COLOR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import candidate_vtln_dirs, load_frame_npy, load_frame_vtln
from grid_transform.source_annotation import load_source_annotation_json
from grid_transform.vt import build_grid, print_grid_summary, visualize_grid


VIS_STYLE_PRESETS = {
    "target": {
        "horiz_color": "#20c997",
        "vert_color": "#ffd166",
        "vt_color": "#ef476f",
        "spine_color": "#118ab2",
        "guide_color": "#ff006e",
        "i_color": "#ef476f",
        "c_color": "#118ab2",
        "p1_color": "#ff006e",
        "m1_color": "#ef476f",
        "l6_color": "#06d6a0",
        "interp_color": "#20c997",
        "mandible_color": "#f4a261",
        "tongue_color": TONGUE_COLOR,
    },
    "source": {
        "horiz_color": "#3a86ff",
        "vert_color": "#ffbe0b",
        "vt_color": "#8338ec",
        "spine_color": "#2a9d8f",
        "guide_color": "#fb5607",
        "i_color": "#8338ec",
        "c_color": "#2a9d8f",
        "p1_color": "#fb5607",
        "m1_color": "#8338ec",
        "l6_color": "#80ed99",
        "interp_color": "#3a86ff",
        "mandible_color": "#e76f51",
        "tongue_color": TONGUE_COLOR,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build vocal-tract grid overlays for one speaker or all available VTLN speakers."
    )
    parser.add_argument(
        "--source",
        choices=("vtln", "nnunet", "annotation"),
        required=True,
        help="Input data source.",
    )
    parser.add_argument(
        "--speaker",
        help="VTLN speaker/image name, for example 1640_s10_0829.",
    )
    parser.add_argument(
        "--all-speakers",
        action="store_true",
        help="When --source vtln is used, render every available VTLN speaker plus one contact sheet.",
    )
    parser.add_argument(
        "--annotation-json",
        type=Path,
        help="Saved edited source annotation JSON to render as a grid source.",
    )
    parser.add_argument(
        "--latest-annotation",
        action="store_true",
        help="Use the newest saved source annotation found under outputs/source_annotation_edits and outputs/annotation_to_grid_transform.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=143020,
        help="nnUNet frame number to load.",
    )
    parser.add_argument(
        "--case",
        default="2008-003^01-1791/test",
        help="nnUNet case path relative to VTLN/data/nnunet_data_80. Groundtruth contours are read from the case's contours folder.",
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Folder containing VTLN images and ROI zip files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=VT_SEG_DATA_ROOT,
        help="Root folder for nnUNet image data.",
    )
    parser.add_argument(
        "--contours-root",
        type=Path,
        default=VT_SEG_CONTOURS_ROOT,
        help="Root folder for nnUNet contour .npy files.",
    )
    parser.add_argument(
        "--n-vert",
        type=int,
        default=9,
        help="Number of vertical grid lines.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=250,
        help="Number of sample points per grid polyline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path for single-speaker mode. Defaults to outputs/<speaker>_grid.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Batch output directory for --all-speakers. Defaults to outputs/speaker_grids.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Do not draw landmark labels on the saved overlay.",
    )
    parser.add_argument(
        "--hide-contours",
        action="store_true",
        help="Do not draw faint contour lines behind the grid.",
    )
    parser.add_argument(
        "--speaker-role",
        choices=("target", "source"),
        help="Visual style preset for the speaker overlay.",
    )
    return parser


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output

    if args.source == "vtln":
        stem = args.speaker
    elif args.source == "annotation":
        stem = resolve_annotation_json_path(args).stem.replace(".latest", "")
    else:
        stem = f"{args.case.replace('/', '_')}_frame_{args.frame}"

    return DEFAULT_OUTPUT_DIR / f"{stem}_grid.png"


def resolve_batch_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return DEFAULT_OUTPUT_DIR / "speaker_grids"


def resolve_speaker_role(args: argparse.Namespace) -> str:
    if args.speaker_role is not None:
        return args.speaker_role
    return "target" if args.source == "vtln" else "source"


def discover_vtln_speakers(vtln_dir: Path) -> list[str]:
    return sorted(path.stem for path in Path(vtln_dir).glob("*.zip"))


def discover_saved_annotation_paths() -> list[Path]:
    patterns = [
        ("source_annotation_edits", "edited_annotation.json"),
        ("annotation_to_grid_transform", "source_annotation.latest.json"),
    ]
    paths: list[Path] = []
    for subdir, filename in patterns:
        root = DEFAULT_OUTPUT_DIR / subdir
        if not root.is_dir():
            continue
        paths.extend(root.rglob(filename))
    return sorted(set(paths))


def resolve_annotation_json_path(args: argparse.Namespace) -> Path:
    if args.annotation_json is not None and args.latest_annotation:
        raise ValueError("Use either --annotation-json or --latest-annotation, not both.")
    if args.annotation_json is not None:
        return args.annotation_json
    if args.latest_annotation:
        candidates = discover_saved_annotation_paths()
        if not candidates:
            raise FileNotFoundError("No saved source annotation JSON files were found.")
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise ValueError("Annotation source requires --annotation-json or --latest-annotation.")


def load_annotation_input(annotation_json: Path) -> tuple[np.ndarray, dict[str, np.ndarray], int, str]:
    payload = load_source_annotation_json(annotation_json)
    metadata = payload["metadata"]

    dataset_root = metadata.get("dataset_root")
    artspeech_speaker = metadata.get("artspeech_speaker")
    session = metadata.get("session")
    source_frame = int(metadata.get("source_frame", 0))
    if not dataset_root or not artspeech_speaker or not session or source_frame <= 0:
        raise ValueError(
            "Saved annotation metadata must include dataset_root, artspeech_speaker, session, and positive source_frame."
        )

    session_data = load_session_data(Path(str(dataset_root)), str(artspeech_speaker), str(session))
    source_frame_index0 = source_frame - 1
    if source_frame_index0 < 0 or source_frame_index0 >= session_data.images.shape[0]:
        raise ValueError(
            f"Saved annotation source_frame {source_frame} is outside session range 1..{session_data.images.shape[0]}."
        )

    image = normalize_frame(
        session_data.images[source_frame_index0],
        session_data.frame_min,
        session_data.frame_max,
    )
    title_name = f"{artspeech_speaker}/{session} frame {source_frame} ({annotation_json.name})"
    return image, payload["contours"], source_frame, title_name


def load_input_data(
    args: argparse.Namespace,
    *,
    speaker_override: str | None = None,
) -> tuple[object, dict[str, np.ndarray], int, str]:
    if args.source == "annotation":
        annotation_json = resolve_annotation_json_path(args)
        return load_annotation_input(annotation_json)

    if args.source == "vtln":
        speaker_id = speaker_override if speaker_override is not None else args.speaker
        if not speaker_id:
            raise ValueError("speaker id is required for VTLN input")
        image, contours = load_frame_vtln(speaker_id, args.vtln_dir)
        return image, contours, 0, speaker_id

    data_dir = args.data_root / args.case
    contours_dir = args.contours_root / args.case
    image, contours = load_frame_npy(args.frame, data_dir, contours_dir)
    return image, contours, args.frame, f"{args.case} frame {args.frame}"


def save_grid_overlay(
    *,
    grid,
    title_name: str,
    speaker_role: str,
    show_contours: bool,
    show_labels: bool,
    output_path: Path,
) -> None:
    style = VIS_STYLE_PRESETS[speaker_role]
    fig = visualize_grid(
        grid,
        figsize=(14, 14),
        show_contours=show_contours,
        show_labels=show_labels,
        style=style,
    )
    ax = fig.axes[0]
    ax.set_title(
        f"{title_name}\nGrid overlay ({speaker_role})",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(image_paths: list[Path], output_path: Path, *, title: str) -> None:
    if not image_paths:
        return

    ncols = min(3, max(1, len(image_paths)))
    nrows = int(math.ceil(len(image_paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5.6 * nrows), dpi=150)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax in axes_arr:
        ax.set_axis_off()

    for ax, image_path in zip(axes_arr, image_paths):
        ax.imshow(plt.imread(image_path))
        stem = image_path.stem
        speaker_name = stem[:-5] if stem.endswith("_grid") else stem
        ax.set_title(speaker_name, fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_and_save_single_grid(
    args: argparse.Namespace,
    *,
    speaker_id: str | None,
    output_path: Path,
) -> None:
    image, contours, frame_number, title_name = load_input_data(args, speaker_override=speaker_id)
    grid = build_grid(
        image,
        contours,
        n_vert=args.n_vert,
        n_points=args.n_points,
        frame_number=frame_number,
    )
    save_grid_overlay(
        grid=grid,
        title_name=title_name,
        speaker_role=resolve_speaker_role(args),
        show_contours=not args.hide_contours,
        show_labels=not args.hide_labels,
        output_path=output_path,
    )
    print_grid_summary(grid)
    print(f"Saved overlay image to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.all_speakers and args.source != "vtln":
        parser.error("--all-speakers is only supported with --source vtln")
    if args.all_speakers and args.output is not None:
        parser.error("--output cannot be used with --all-speakers; use --output-dir instead")
    if not args.all_speakers and args.output_dir is not None:
        parser.error("--output-dir is only used with --all-speakers")
    if args.source == "vtln" and not args.speaker and not args.all_speakers:
        parser.error("--speaker is required when --source vtln")
    if args.source != "annotation" and (args.annotation_json is not None or args.latest_annotation):
        parser.error("--annotation-json/--latest-annotation require --source annotation")
    if args.source == "annotation" and args.all_speakers:
        parser.error("--all-speakers is not supported with --source annotation")
    if args.source == "annotation" and args.speaker is not None:
        parser.error("--speaker is not used with --source annotation")

    if not args.all_speakers:
        build_and_save_single_grid(args, speaker_id=None, output_path=resolve_output_path(args))
        return 0

    speaker_ids = discover_vtln_speakers(args.vtln_dir)
    if not speaker_ids:
        raise FileNotFoundError(f"No VTLN ROI zip files found in {args.vtln_dir}")

    output_dir = resolve_batch_output_dir(args)
    speaker_output_dir = output_dir / "speakers"
    saved_paths: list[Path] = []
    failures: list[tuple[str, str]] = []

    for speaker_id in speaker_ids:
        output_path = speaker_output_dir / f"{speaker_id}_grid.png"
        try:
            build_and_save_single_grid(args, speaker_id=speaker_id, output_path=output_path)
            saved_paths.append(output_path)
        except Exception as exc:
            failures.append((speaker_id, f"{type(exc).__name__}: {exc}"))
            print(f"Failed speaker {speaker_id}: {type(exc).__name__}: {exc}")

    if not saved_paths:
        raise RuntimeError("No speaker grids were rendered successfully.")

    contact_sheet_path = output_dir / "all_speakers_grid_contact_sheet.png"
    save_contact_sheet(saved_paths, contact_sheet_path, title="All VTLN Speaker Grids")
    print(f"Saved contact sheet to: {contact_sheet_path}")

    if failures:
        print("Failed speakers:")
        for speaker_id, message in failures:
            print(f"  {speaker_id}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
