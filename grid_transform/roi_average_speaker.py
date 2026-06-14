from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import numpy as np
from PIL import Image, ImageDraw

from grid_transform.analysis_shared import (
    CURATED_SPEAKER_GENDER,
    LoadedSpeaker,
    choose_label_colors,
    load_curated_speakers,
    read_curated_specs_map,
    speaker_id_sort_key,
)
from grid_transform.transform_helpers import (
    apply_transform,
    build_step1_anchors,
    estimate_affine,
    extract_true_landmarks,
    format_target_frame,
)


DEFAULT_EXCLUDE_LABELS = (
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "lower-lip",
)


def default_curated_speaker_ids(vtln_dir: Path) -> list[str]:
    """Return curated speaker ids from the manifest in stable P-order."""
    specs = read_curated_specs_map(vtln_dir)
    speaker_ids = [speaker_id for speaker_id in specs if speaker_id in CURATED_SPEAKER_GENDER]
    return sorted(speaker_ids, key=speaker_id_sort_key)


def load_roi_average_speakers(vtln_dir: Path, requested_speakers: list[str] | None = None) -> dict[str, LoadedSpeaker]:
    """Load the curated speaker subset used by the ROI average-speaker workflow."""
    speaker_ids = requested_speakers or default_curated_speaker_ids(vtln_dir)
    return load_curated_speakers(vtln_dir, speaker_ids)


def resolve_roi_average_labels(
    speakers: dict[str, LoadedSpeaker],
    exclude_labels: Collection[str] = DEFAULT_EXCLUDE_LABELS,
) -> tuple[list[str], list[dict[str, object]]]:
    """Pick globally shared labels suitable for ROI-overlap scoring."""
    exclude_set = {label.strip() for label in exclude_labels if label.strip()}
    if not speakers:
        raise ValueError("No speakers were loaded.")

    all_labels = sorted({label for speaker in speakers.values() for label in speaker.contours})
    ordered_speakers = sorted(speakers, key=speaker_id_sort_key)
    included_labels: list[str] = []
    excluded_rows: list[dict[str, object]] = []

    for label in all_labels:
        reasons: list[dict[str, object]] = []
        if label in exclude_set:
            reasons.append({"type": "explicit_exclude"})

        missing_speakers = [speaker_id for speaker_id in ordered_speakers if label not in speakers[speaker_id].contours]
        if missing_speakers:
            reasons.append({"type": "missing_from_global_intersection", "speakers": missing_speakers})

        short_speakers = [
            speaker_id
            for speaker_id in ordered_speakers
            if label in speakers[speaker_id].contours and len(np.asarray(speakers[speaker_id].contours[label], dtype=float)) < 3
        ]
        if short_speakers:
            reasons.append({"type": "too_few_points", "speakers": short_speakers})

        if reasons:
            excluded_rows.append({"label": label, "reasons": reasons})
        else:
            included_labels.append(label)

    if not included_labels:
        raise ValueError("No ROI-overlap labels remained after applying exclusions and global-sharing checks.")
    return included_labels, excluded_rows


def build_affine_only_transform(source_grid, target_grid) -> dict[str, object]:
    """Fit the Step-1 affine-only transform using the current shared anchor set."""
    lm_src = extract_true_landmarks(source_grid)
    lm_tgt = extract_true_landmarks(target_grid)
    step1_src, step1_tgt, step1_labels = build_step1_anchors(lm_src, lm_tgt)
    affine = estimate_affine(step1_src, step1_tgt)

    def apply_affine(points: np.ndarray) -> np.ndarray:
        return np.asarray(apply_transform(affine, np.asarray(points, dtype=float)), dtype=float)

    return {
        "step1_affine": affine,
        "step1_labels": list(step1_labels),
        "apply_affine": apply_affine,
    }


def close_contour_polygon(points: np.ndarray) -> np.ndarray:
    """Close an open contour with a straight last->first segment."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected contour points with shape (N, 2), got {pts.shape}.")
    if len(pts) < 3:
        raise ValueError(f"Need at least 3 contour points to build a polygon, got {len(pts)}.")
    if np.allclose(pts[0], pts[-1]):
        return pts.copy()
    return np.vstack([pts, pts[0]])


def polygon_to_mask(points: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Rasterize a polygon contour into a binary ROI mask."""
    polygon = close_contour_polygon(points)
    height, width = int(image_shape[0]), int(image_shape[1])
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height - 1))
    xy = [tuple(int(value) for value in pair) for pair in np.rint(clipped)]

    mask_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(xy, outline=1, fill=1)
    return np.asarray(mask_image, dtype=bool)


def compute_mask_overlap_metrics(source_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float | int]:
    """Compute Dice/IoU overlap statistics from two binary masks."""
    src = np.asarray(source_mask, dtype=bool)
    tgt = np.asarray(target_mask, dtype=bool)
    intersection = int(np.count_nonzero(src & tgt))
    union = int(np.count_nonzero(src | tgt))
    source_area = int(np.count_nonzero(src))
    target_area = int(np.count_nonzero(tgt))

    denom = source_area + target_area
    dice = 1.0 if denom == 0 else float((2.0 * intersection) / denom)
    iou = 1.0 if union == 0 else float(intersection / union)
    return {
        "dice": dice,
        "iou": iou,
        "intersection": intersection,
        "union": union,
        "source_area": source_area,
        "target_area": target_area,
    }


@dataclass(frozen=True)
class RoiLabelScore:
    label: str
    dice: float
    iou: float
    intersection: int
    union: int
    source_area: int
    target_area: int

    def to_row(self, pair: "AffineRoiPairMetrics") -> dict[str, object]:
        return {
            "source_speaker": pair.source_speaker_id,
            "source_basename": pair.source_basename,
            "source_gender": pair.source_gender,
            "target_speaker": pair.target_speaker_id,
            "target_basename": pair.target_basename,
            "target_gender": pair.target_gender,
            "label": self.label,
            "dice": self.dice,
            "iou": self.iou,
            "intersection": self.intersection,
            "union": self.union,
            "source_area": self.source_area,
            "target_area": self.target_area,
        }


@dataclass
class AffineRoiPairMetrics:
    source_speaker_id: str
    source_basename: str
    source_gender: str
    target_speaker_id: str
    target_basename: str
    target_gender: str
    labels_used: list[str]
    step1_labels: list[str]
    label_scores: list[RoiLabelScore]
    mapped_contours: dict[str, np.ndarray]

    @property
    def pair_mean_dice(self) -> float:
        return float(np.mean([score.dice for score in self.label_scores]))

    @property
    def pair_mean_iou(self) -> float:
        return float(np.mean([score.iou for score in self.label_scores]))

    def to_row(self) -> dict[str, object]:
        return {
            "source_speaker": self.source_speaker_id,
            "source_basename": self.source_basename,
            "source_gender": self.source_gender,
            "target_speaker": self.target_speaker_id,
            "target_basename": self.target_basename,
            "target_gender": self.target_gender,
            "pair_mean_dice": self.pair_mean_dice,
            "pair_mean_iou": self.pair_mean_iou,
            "label_count": len(self.labels_used),
            "labels_used": ",".join(self.labels_used),
            "step1_labels": ",".join(self.step1_labels),
        }


@dataclass
class RoiAverageSpeakerCandidate:
    target_speaker_id: str
    target_basename: str
    target_gender: str
    pair_results: list[AffineRoiPairMetrics]
    rank: int = 0

    @property
    def mean_pair_dice(self) -> float:
        return float(np.mean([pair.pair_mean_dice for pair in self.pair_results]))

    @property
    def mean_pair_iou(self) -> float:
        return float(np.mean([pair.pair_mean_iou for pair in self.pair_results]))

    @property
    def std_pair_dice(self) -> float:
        return float(np.std([pair.pair_mean_dice for pair in self.pair_results]))

    @property
    def min_pair_dice(self) -> float:
        return float(np.min([pair.pair_mean_dice for pair in self.pair_results]))

    @property
    def max_pair_dice(self) -> float:
        return float(np.max([pair.pair_mean_dice for pair in self.pair_results]))

    def to_row(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "target_speaker": self.target_speaker_id,
            "target_basename": self.target_basename,
            "target_gender": self.target_gender,
            "pair_count": len(self.pair_results),
            "mean_pair_dice": self.mean_pair_dice,
            "mean_pair_iou": self.mean_pair_iou,
            "std_pair_dice": self.std_pair_dice,
            "min_pair_dice": self.min_pair_dice,
            "max_pair_dice": self.max_pair_dice,
        }


def compute_affine_roi_pair_metrics(
    source: LoadedSpeaker,
    target: LoadedSpeaker,
    labels: list[str],
) -> AffineRoiPairMetrics:
    """Compute affine-only ROI overlap metrics for one ordered source->target pair."""
    transform = build_affine_only_transform(source.grid, target.grid)
    mapped_contours = {
        label: transform["apply_affine"](np.asarray(source.contours[label], dtype=float))
        for label in labels
    }
    image_shape = np.asarray(target.image).shape

    label_scores: list[RoiLabelScore] = []
    for label in labels:
        source_mask = polygon_to_mask(mapped_contours[label], image_shape)
        target_mask = polygon_to_mask(np.asarray(target.contours[label], dtype=float), image_shape)
        metrics = compute_mask_overlap_metrics(source_mask, target_mask)
        label_scores.append(
            RoiLabelScore(
                label=label,
                dice=float(metrics["dice"]),
                iou=float(metrics["iou"]),
                intersection=int(metrics["intersection"]),
                union=int(metrics["union"]),
                source_area=int(metrics["source_area"]),
                target_area=int(metrics["target_area"]),
            )
        )

    return AffineRoiPairMetrics(
        source_speaker_id=source.spec.speaker_id,
        source_basename=source.spec.basename,
        source_gender=source.spec.gender,
        target_speaker_id=target.spec.speaker_id,
        target_basename=target.spec.basename,
        target_gender=target.spec.gender,
        labels_used=list(labels),
        step1_labels=list(transform["step1_labels"]),
        label_scores=label_scores,
        mapped_contours=mapped_contours,
    )


def rank_affine_roi_average_speakers(
    speakers: dict[str, LoadedSpeaker],
    labels: list[str],
) -> dict[str, object]:
    """Score every curated speaker as a target and rank the best affine-only ROI average-speaker candidate."""
    ordered_ids = sorted(speakers, key=speaker_id_sort_key)
    pair_results: list[AffineRoiPairMetrics] = []
    candidates: list[RoiAverageSpeakerCandidate] = []

    for target_id in ordered_ids:
        target = speakers[target_id]
        target_pairs: list[AffineRoiPairMetrics] = []
        for source_id in ordered_ids:
            if source_id == target_id:
                continue
            pair = compute_affine_roi_pair_metrics(speakers[source_id], target, labels)
            target_pairs.append(pair)
            pair_results.append(pair)

        candidates.append(
            RoiAverageSpeakerCandidate(
                target_speaker_id=target.spec.speaker_id,
                target_basename=target.spec.basename,
                target_gender=target.spec.gender,
                pair_results=target_pairs,
            )
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.mean_pair_dice,
            -candidate.mean_pair_iou,
            candidate.std_pair_dice,
            speaker_id_sort_key(candidate.target_speaker_id),
        ),
    )
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate.rank = index

    return {
        "labels_used": list(labels),
        "pair_results": pair_results,
        "candidates": ranked_candidates,
        "winner": ranked_candidates[0],
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _serialize_label_summary(candidate: RoiAverageSpeakerCandidate, labels: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in labels:
        label_pairs = [pair for pair in candidate.pair_results if label in pair.labels_used]
        dice_values = [
            next(score.dice for score in pair.label_scores if score.label == label)
            for pair in label_pairs
        ]
        iou_values = [
            next(score.iou for score in pair.label_scores if score.label == label)
            for pair in label_pairs
        ]
        rows.append(
            {
                "label": label,
                "pair_count": len(label_pairs),
                "mean_dice": float(np.mean(dice_values)),
                "mean_iou": float(np.mean(iou_values)),
                "min_dice": float(np.min(dice_values)),
                "max_dice": float(np.max(dice_values)),
            }
        )
    return rows


def _serialize_pair_rows(candidates: list[RoiAverageSpeakerCandidate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        for pair in candidate.pair_results:
            row = pair.to_row()
            row["target_rank"] = candidate.rank
            rows.append(row)
    return rows


def _serialize_label_rows(pair_results: list[AffineRoiPairMetrics]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in pair_results:
        rows.extend(score.to_row(pair) for score in pair.label_scores)
    return rows


def _build_overlap_preview(pair: AffineRoiPairMetrics, target: LoadedSpeaker) -> np.ndarray:
    image_shape = np.asarray(target.image).shape
    source_union = np.zeros(image_shape[:2], dtype=bool)
    target_union = np.zeros(image_shape[:2], dtype=bool)
    for label in pair.labels_used:
        source_union |= polygon_to_mask(pair.mapped_contours[label], image_shape)
        target_union |= polygon_to_mask(np.asarray(target.contours[label], dtype=float), image_shape)

    overlap = source_union & target_union
    source_only = source_union & ~target_union
    target_only = target_union & ~source_union

    preview = np.ones((image_shape[0], image_shape[1], 3), dtype=float)
    preview[target_only] = np.array([0.70, 0.86, 1.00], dtype=float)
    preview[source_only] = np.array([1.00, 0.78, 0.82], dtype=float)
    preview[overlap] = np.array([0.67, 0.92, 0.72], dtype=float)
    return preview


def _save_pair_figure(
    pair: AffineRoiPairMetrics,
    source: LoadedSpeaker,
    target: LoadedSpeaker,
    label_colors: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.ravel()

    format_target_frame(axes[0], source.image, f"Source speaker: {pair.source_basename}")
    for label in pair.labels_used:
        pts = np.asarray(source.contours[label], dtype=float)
        axes[0].plot(pts[:, 0], pts[:, 1], color=label_colors[label], lw=2.1, alpha=0.95)

    format_target_frame(axes[1], target.image, f"Target speaker: {pair.target_basename}")
    for label in pair.labels_used:
        pts = np.asarray(target.contours[label], dtype=float)
        axes[1].plot(pts[:, 0], pts[:, 1], color=label_colors[label], lw=2.1, alpha=0.95)

    format_target_frame(axes[2], target.image, "Affine-mapped source vs target")
    for label in pair.labels_used:
        target_pts = np.asarray(target.contours[label], dtype=float)
        mapped_pts = np.asarray(pair.mapped_contours[label], dtype=float)
        axes[2].plot(target_pts[:, 0], target_pts[:, 1], color=label_colors[label], lw=2.0, alpha=0.45)
        axes[2].plot(mapped_pts[:, 0], mapped_pts[:, 1], color=label_colors[label], lw=2.2, alpha=0.98, linestyle="--")
    axes[2].text(
        0.02,
        0.02,
        "\n".join(
            [
                f"Pair mean Dice: {pair.pair_mean_dice:.4f}",
                f"Pair mean IoU: {pair.pair_mean_iou:.4f}",
                f"Labels used: {len(pair.labels_used)}",
                "solid = target, dashed = affine-mapped source",
            ]
        ),
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.94),
    )

    overlap_preview = _build_overlap_preview(pair, target)
    axes[3].imshow(overlap_preview)
    axes[3].set_title("ROI overlap preview", fontsize=13, fontweight="bold")
    axes[3].axis("off")
    lines = ["label                    Dice    IoU"]
    for score in pair.label_scores:
        lines.append(f"{score.label:22s} {score.dice:5.3f}  {score.iou:5.3f}")
    axes[3].text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=axes[3].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.96),
    )

    fig.suptitle(
        f"{pair.source_basename} -> {pair.target_basename} | affine-only ROI Dice",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_target_overview_figure(
    candidate: RoiAverageSpeakerCandidate,
    target: LoadedSpeaker,
    labels: list[str],
    label_colors: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(23, 7.8))

    format_target_frame(
        axes[0],
        target.image,
        f"Target {candidate.target_basename} | all affine-mapped sources",
    )
    for label in labels:
        target_pts = np.asarray(target.contours[label], dtype=float)
        axes[0].plot(target_pts[:, 0], target_pts[:, 1], color=label_colors[label], lw=2.2, alpha=0.72)
    for pair in candidate.pair_results:
        for label in labels:
            mapped_pts = np.asarray(pair.mapped_contours[label], dtype=float)
            axes[0].plot(mapped_pts[:, 0], mapped_pts[:, 1], color=label_colors[label], lw=1.2, alpha=0.18, linestyle="--")
    axes[0].text(
        0.02,
        0.02,
        "\n".join(
            [
                f"Rank: {candidate.rank}",
                f"Mean Dice: {candidate.mean_pair_dice:.4f}",
                f"Mean IoU: {candidate.mean_pair_iou:.4f}",
                f"Sources mapped: {len(candidate.pair_results)}",
            ]
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.94),
    )

    sorted_pairs = sorted(
        candidate.pair_results,
        key=lambda pair: (-pair.pair_mean_dice, speaker_id_sort_key(pair.source_speaker_id)),
    )
    pair_labels = [pair.source_speaker_id for pair in sorted_pairs]
    pair_values = [pair.pair_mean_dice for pair in sorted_pairs]
    y_pos = np.arange(len(sorted_pairs))
    axes[1].barh(y_pos, pair_values, color="#2a9d8f", edgecolor="black", alpha=0.84)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(pair_labels, fontsize=9)
    axes[1].set_xlabel("Pair mean Dice")
    axes[1].set_title("Ordered source -> target Dice", fontsize=13, fontweight="bold")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].invert_yaxis()
    axes[1].grid(True, axis="x", alpha=0.25)
    for yi, val in zip(y_pos, pair_values):
        axes[1].text(min(val + 0.01, 0.995), yi, f"{val:.3f}", va="center", fontsize=8.5)

    label_summary = _serialize_label_summary(candidate, labels)
    label_summary = sorted(label_summary, key=lambda row: (-float(row["mean_dice"]), str(row["label"])))
    label_names = [row["label"] for row in label_summary]
    label_values = [float(row["mean_dice"]) for row in label_summary]
    bar_colors = [label_colors[label] for label in label_names]
    y_pos = np.arange(len(label_summary))
    axes[2].barh(y_pos, label_values, color=bar_colors, edgecolor="black", alpha=0.86)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(label_names, fontsize=9)
    axes[2].set_xlabel("Mean label Dice")
    axes[2].set_title("Label mean Dice for this target", fontsize=13, fontweight="bold")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].invert_yaxis()
    axes[2].grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        f"{candidate.target_basename} ({candidate.target_speaker_id}) | ROI average-speaker target overview",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_affine_roi_average_speaker_report(
    speakers: dict[str, LoadedSpeaker],
    ranking: dict[str, object],
    output_dir: Path,
    *,
    config: dict[str, object] | None = None,
    excluded_labels: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Write CSV/JSON/Markdown reports plus pair/target figures for the ROI average-speaker analysis."""
    labels = list(ranking["labels_used"])
    candidates: list[RoiAverageSpeakerCandidate] = list(ranking["candidates"])
    pair_results: list[AffineRoiPairMetrics] = list(ranking["pair_results"])
    winner: RoiAverageSpeakerCandidate = ranking["winner"]

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = output_dir / "pairs"
    targets_dir = output_dir / "targets"
    target_details_dir = output_dir / "target_details"
    target_details_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    candidate_scores_path = output_dir / "candidate_scores.csv"
    pair_scores_path = output_dir / "pair_scores.csv"
    label_scores_path = output_dir / "label_scores.csv"

    candidate_rows = [candidate.to_row() for candidate in candidates]
    pair_rows = _serialize_pair_rows(candidates)
    label_rows = _serialize_label_rows(pair_results)

    _write_csv(
        candidate_scores_path,
        [
            "rank",
            "target_speaker",
            "target_basename",
            "target_gender",
            "pair_count",
            "mean_pair_dice",
            "mean_pair_iou",
            "std_pair_dice",
            "min_pair_dice",
            "max_pair_dice",
        ],
        candidate_rows,
    )
    _write_csv(
        pair_scores_path,
        [
            "target_rank",
            "source_speaker",
            "source_basename",
            "source_gender",
            "target_speaker",
            "target_basename",
            "target_gender",
            "pair_mean_dice",
            "pair_mean_iou",
            "label_count",
            "labels_used",
            "step1_labels",
        ],
        pair_rows,
    )
    _write_csv(
        label_scores_path,
        [
            "source_speaker",
            "source_basename",
            "source_gender",
            "target_speaker",
            "target_basename",
            "target_gender",
            "label",
            "dice",
            "iou",
            "intersection",
            "union",
            "source_area",
            "target_area",
        ],
        label_rows,
    )

    label_colors = choose_label_colors(labels)
    for candidate in candidates:
        target = speakers[candidate.target_speaker_id]
        _save_target_overview_figure(
            candidate,
            target,
            labels,
            label_colors,
            targets_dir / f"{candidate.target_basename}_overview.png",
        )

        detail_payload = {
            "target": candidate.to_row(),
            "label_summary": _serialize_label_summary(candidate, labels),
            "pairs": [pair.to_row() for pair in candidate.pair_results],
        }
        (target_details_dir / f"{candidate.target_basename}.json").write_text(
            json.dumps(detail_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for pair in candidate.pair_results:
            _save_pair_figure(
                pair,
                speakers[pair.source_speaker_id],
                target,
                label_colors,
                pairs_dir / f"{pair.source_basename}_to_{pair.target_basename}.png",
            )

    top_rows = candidate_rows[:3]
    bottom_rows = candidate_rows[-3:]
    summary_payload = {
        "config": config or {},
        "speakers": [
            {
                "speaker": speaker_id,
                "basename": speakers[speaker_id].spec.basename,
                "gender": speakers[speaker_id].spec.gender,
            }
            for speaker_id in sorted(speakers, key=speaker_id_sort_key)
        ],
        "labels_used": labels,
        "excluded_labels": excluded_labels or [],
        "winner": winner.to_row(),
        "ranking": candidate_rows,
        "paths": {
            "candidate_scores_csv": str(candidate_scores_path),
            "pair_scores_csv": str(pair_scores_path),
            "label_scores_csv": str(label_scores_path),
            "targets_dir": str(targets_dir),
            "pairs_dir": str(pairs_dir),
            "target_details_dir": str(target_details_dir),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ROI-Dice Affine Average Speaker Summary",
        "",
        "Primary metric: Dice overlap on affine-mapped closed ROI polygons.",
        "Secondary metric: IoU overlap on the same ROI masks.",
        "",
        f"- Winner: `{winner.target_speaker_id}` / `{winner.target_basename}`",
        f"- Mean pair Dice: `{winner.mean_pair_dice:.4f}`",
        f"- Mean pair IoU: `{winner.mean_pair_iou:.4f}`",
        f"- Labels used ({len(labels)}): {', '.join(labels)}",
        "",
        "## Top candidates",
        "",
        "| rank | speaker | basename | mean Dice | mean IoU | std Dice |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in top_rows:
        lines.append(
            f"| {row['rank']} | {row['target_speaker']} | {row['target_basename']} | "
            f"{row['mean_pair_dice']:.4f} | {row['mean_pair_iou']:.4f} | {row['std_pair_dice']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Lowest candidates",
            "",
            "| rank | speaker | basename | mean Dice | mean IoU | std Dice |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in bottom_rows:
        lines.append(
            f"| {row['rank']} | {row['target_speaker']} | {row['target_basename']} | "
            f"{row['mean_pair_dice']:.4f} | {row['mean_pair_iou']:.4f} | {row['std_pair_dice']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Ordered pairs use `source != target` only.",
            "- Affine-only means only Step 1 anchors are used; no TPS refinement is applied.",
            "- Every included label contributes equally to the pair score.",
        ]
    )
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "candidate_scores_csv": candidate_scores_path,
        "pair_scores_csv": pair_scores_path,
        "label_scores_csv": label_scores_path,
        "pairs_dir": pairs_dir,
        "targets_dir": targets_dir,
        "target_details_dir": target_details_dir,
    }


__all__ = [
    "DEFAULT_EXCLUDE_LABELS",
    "default_curated_speaker_ids",
    "load_roi_average_speakers",
    "resolve_roi_average_labels",
    "build_affine_only_transform",
    "close_contour_polygon",
    "polygon_to_mask",
    "compute_mask_overlap_metrics",
    "compute_affine_roi_pair_metrics",
    "rank_affine_roi_average_speakers",
    "write_affine_roi_average_speaker_report",
]
