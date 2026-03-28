from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.artspeech_video import (
    LabelSnapshot,
    build_label_snapshots,
    load_session_data,
    normalized_session_frame,
    resolve_default_dataset_root,
)
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR, TONGUE_COLOR
from grid_transform.io import load_frame_vtln


REFERENCE_CONTOUR_COLORS = {
    "tongue": TONGUE_COLOR,
    "incisior-hard-palate": "#ef476f",
    "soft-palate-midline": "#8338ec",
    "soft-palate": "#ff7f50",
    "mandible-incisior": "#f4a261",
    "pharynx": "#118ab2",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one VTLN reference speaker with labeled snapshots from an ArtSpeech session."
    )
    parser.add_argument("--vtln-speaker", default="1640_s10_0829", help="VTLN speaker/image name.")
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR, help="Folder containing VTLN images and ROI zips.")
    parser.add_argument("--artspeech-speaker", default="P7", help="ArtSpeech speaker id, for example P7.")
    parser.add_argument("--session", default="S10", help="ArtSpeech session id, for example S10.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional explicit ArtSpeech dataset root. Defaults to an auto-detected local path.",
    )
    parser.add_argument("--max-samples", type=int, default=6, help="Number of labeled ArtSpeech snapshots to show.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Defaults to outputs/comparisons/<vtln>_vs_<speaker>_<session>_labels.png.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional output JSON path. Defaults next to the PNG.",
    )
    return parser.parse_args(argv)


def default_output_path(vtln_speaker: str, artspeech_speaker: str, session: str) -> Path:
    return DEFAULT_OUTPUT_DIR / "comparisons" / f"{vtln_speaker}_vs_{artspeech_speaker}_{session}_labels.png"


def wrap_sentence(text: str, width: int = 42) -> str:
    clean = " ".join(text.split())
    if not clean:
        return "-"
    return textwrap.fill(clean, width=width)


def contour_anchor(points: np.ndarray) -> np.ndarray:
    return np.asarray(points[len(points) // 2], dtype=float)


def draw_reference_panel(ax, image, contours: dict[str, np.ndarray], speaker_name: str) -> None:
    ax.imshow(image, cmap="gray")
    ax.set_title(f"VTLN reference\n{speaker_name}", fontsize=12, fontweight="bold")
    ax.axis("off")

    for name, points in sorted(contours.items()):
        pts = np.asarray(points, dtype=float)
        color = REFERENCE_CONTOUR_COLORS.get(name, "#00b4d8")
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.8, alpha=0.92)
        anchor = contour_anchor(pts)
        ax.text(
            anchor[0],
            anchor[1],
            name,
            color=color,
            fontsize=7,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.72, ec="none"),
        )


def draw_snapshot_panel(ax, frame: np.ndarray, snapshot: LabelSnapshot, artspeech_speaker: str, session: str) -> None:
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"{artspeech_speaker}/{session}  frame {snapshot.frame_index + 1}\n{snapshot.word or '-'}",
        fontsize=10,
        fontweight="bold",
    )
    ax.axis("off")
    text = "\n".join(
        [
            f"time: {snapshot.time_s:.3f}s",
            f"word: {snapshot.word or '-'}",
            f"phoneme: {snapshot.phoneme or '-'}",
            f"sentence: {wrap_sentence(snapshot.sentence)}",
        ]
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        family="monospace",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7, ec="none"),
    )


def build_metadata(
    vtln_speaker: str,
    artspeech_speaker: str,
    session: str,
    dataset_root: Path,
    snapshots: list[LabelSnapshot],
) -> dict[str, object]:
    return {
        "vtln_speaker": vtln_speaker,
        "artspeech_speaker": artspeech_speaker,
        "session": session,
        "dataset_root": str(dataset_root),
        "snapshots": [
            {
                "frame_index": snapshot.frame_index,
                "time_sec": snapshot.time_s,
                "word": snapshot.word,
                "phoneme": snapshot.phoneme,
                "sentence": snapshot.sentence,
            }
            for snapshot in snapshots
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_root = args.dataset_root or resolve_default_dataset_root(args.artspeech_speaker)
    output_path = args.output or default_output_path(args.vtln_speaker, args.artspeech_speaker, args.session)
    metadata_path = args.metadata_output or output_path.with_suffix(".json")

    print("[load] reading VTLN reference")
    reference_image, reference_contours = load_frame_vtln(args.vtln_speaker, args.vtln_dir)

    print("[load] reading ArtSpeech session")
    session_data = load_session_data(dataset_root, args.artspeech_speaker, args.session)
    snapshots = build_label_snapshots(session_data, max_samples=args.max_samples)
    if not snapshots:
        raise ValueError("No non-empty labeled intervals were found in the first TextGrid tier.")

    ncols = 3
    nrows = int(np.ceil(len(snapshots) / ncols))
    fig = plt.figure(figsize=(6 + 4.6 * ncols, 4.8 * max(nrows, 1)))
    layout = fig.add_gridspec(nrows=max(nrows, 1), ncols=ncols + 1, width_ratios=[1.35, 1.0, 1.0, 1.0])

    reference_ax = fig.add_subplot(layout[:, 0])
    draw_reference_panel(reference_ax, reference_image, reference_contours, args.vtln_speaker)

    snapshot_axes = []
    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols + 1
        snapshot_axes.append(fig.add_subplot(layout[row, col]))

    for ax, snapshot in zip(snapshot_axes, snapshots):
        frame = normalized_session_frame(session_data, snapshot.frame_index)
        draw_snapshot_panel(ax, frame, snapshot, args.artspeech_speaker, args.session)

    for ax in snapshot_axes[len(snapshots) :]:
        ax.axis("off")

    fig.suptitle(
        f"VTLN {args.vtln_speaker} vs labeled ArtSpeech snapshots ({args.artspeech_speaker}/{args.session})",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = build_metadata(args.vtln_speaker, args.artspeech_speaker, args.session, dataset_root, snapshots)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"comparison_figure": str(output_path), "metadata_json": str(metadata_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
