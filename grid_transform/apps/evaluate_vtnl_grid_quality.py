from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTNL_DIR
from grid_transform.io import load_frame_vtnl
from grid_transform.vt import GridValidationError, build_grid, validate_grid_contours, visualize_grid


DEFAULT_OUTPUT_DIRNAME = DEFAULT_OUTPUT_DIR / "vtnl_grid_diagnostics"
ADJACENT_SPACING_WARNING_PX = 10.0
H1_CLAMP_WARNING_RATIO = 0.20
AUDIT_STYLE = {
    "horiz_color": "#20c997",
    "vert_color": "#ffd166",
    "vt_color": "#ef476f",
    "spine_color": "#118ab2",
    "guide_color": "#ff006e",
    "i_color": "#ef476f",
    "i_label_color": "#ef476f",
    "i_label_fc": "white",
    "c_color": "#118ab2",
    "c_label_color": "#118ab2",
    "c_label_fc": "white",
    "p1_color": "#ff006e",
    "p1_label_color": "#ff006e",
    "p1_label_fc": "white",
    "m1_color": "#ef476f",
    "m1_label_color": "#ef476f",
    "m1_label_fc": "white",
    "l6_color": "#06d6a0",
    "l6_label_color": "#06d6a0",
    "l6_label_fc": "black",
    "interp_color": "#20c997",
    "interp_label_color": "#20c997",
    "interp_label_fc": "black",
    "mandible_color": "#f4a261",
    "tongue_color": "#ff006e",
}
PATTERN_GUIDANCE = {
    "missing_required_contour": "Inspect the ROI zip for missing hard-required contours or broken contour naming.",
    "sparse_core_contour": "Revisit sparse landmarks/ROIs; the grid may build but anchor geometry is likely unstable.",
    "tight_spacing": "Inspect adjacent grid lines for collapse or over-bending; consider softening the H1 tail or revisiting landmarks.",
    "line_crossing": "This is a topology warning; check whether H or V lines fold over each other.",
    "high_H1_clamp": "The H1 tail is deviating strongly from the I5->C1 chord; the speaker may need a looser or smarter tail model.",
    "missing_P1": "No usable pharynx intersection was found for the H1 tail; inspect pharynx and soft-palate geometry.",
    "out_of_bounds": "Some grid points leave the image bounds; inspect landmark extraction and H1 construction.",
}


@dataclass
class SpeakerAuditResult:
    speaker: str
    status: str
    hard_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    contour_point_counts: dict[str, int] = field(default_factory=dict)
    landmark_presence: dict[str, bool] = field(default_factory=dict)
    out_of_bounds_points: int = 0
    crossing_horiz_pairs: int = 0
    crossing_vert_pairs: int = 0
    adjacent_horiz_min_spacing: float | None = None
    adjacent_vert_min_spacing: float | None = None
    h1_dev_ratio: float | None = None
    h1_max_clamp: float | None = None
    output_image: str = ""
    recommendations: list[str] = field(default_factory=list)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def hard_error_count(self) -> int:
        return len(self.hard_errors)

    @property
    def is_issue(self) -> bool:
        return self.status != "ok" or self.warning_count > 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit VTNL vocal-tract grids for every speaker in a folder, "
            "saving per-speaker overlays plus summary JSON/CSV/Markdown reports."
        )
    )
    parser.add_argument(
        "--vtnl-dir",
        type=Path,
        default=DEFAULT_VTNL_DIR,
        help="Folder containing VTNL images and ROI zip files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRNAME,
        help="Where to save audit figures and summary files.",
    )
    parser.add_argument("--n-vert", type=int, default=9, help="Number of vertical grid lines.")
    parser.add_argument("--n-points", type=int, default=250, help="Number of points per grid polyline.")
    parser.add_argument(
        "--max-dev-ratio",
        type=float,
        default=0.25,
        help="Maximum allowed perpendicular deviation ratio for the H1 I5->C1 tail.",
    )
    return parser.parse_args(argv)


def discover_vtnl_speakers(vtnl_dir: Path) -> list[str]:
    return sorted(path.stem for path in Path(vtnl_dir).glob("*.zip"))


def parse_issue_code(message: str) -> str:
    return message.split(":", 1)[0].strip() if ":" in message else message.strip()


def adjacent_min_spacing(lines: list[np.ndarray]) -> float | None:
    if len(lines) < 2:
        return None
    minima = []
    for first, second in zip(lines[:-1], lines[1:]):
        distances = np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float), axis=1)
        minima.append(float(np.min(distances)))
    return min(minima) if minima else None


def count_crossing_pairs(lines: list[np.ndarray]) -> int:
    try:
        from shapely.geometry import LineString
    except Exception:
        return 0

    crossings = 0
    for i in range(len(lines)):
        first = LineString(np.asarray(lines[i], dtype=float))
        for j in range(i + 1, len(lines)):
            second = LineString(np.asarray(lines[j], dtype=float))
            if first.crosses(second):
                crossings += 1
    return crossings


def count_out_of_bounds_points(lines: list[np.ndarray], image_shape: tuple[int, int]) -> int:
    if not lines:
        return 0
    height, width = image_shape
    pts = np.vstack([np.asarray(line, dtype=float) for line in lines])
    return int(
        np.count_nonzero(
            (pts[:, 0] < 0)
            | (pts[:, 0] > width - 1)
            | (pts[:, 1] < 0)
            | (pts[:, 1] > height - 1)
        )
    )


def landmark_presence_from_grid(grid: Any) -> dict[str, bool]:
    presence = {f"I{i}": False for i in range(1, 8)}
    if grid.I_points:
        for key in presence:
            presence[key] = key in grid.I_points and grid.I_points[key] is not None
    presence["M1"] = grid.M1 is not None
    presence["L6"] = grid.L6 is not None
    presence["P1"] = grid.P1_point is not None
    return presence


def add_geometry_warnings(result: SpeakerAuditResult) -> None:
    if result.out_of_bounds_points > 0:
        result.warnings.append(
            f"out_of_bounds: {result.out_of_bounds_points} grid points fall outside the image bounds"
        )
    total_crossings = result.crossing_horiz_pairs + result.crossing_vert_pairs
    if total_crossings > 0:
        result.warnings.append(
            f"line_crossing: {result.crossing_horiz_pairs} horizontal pair(s), "
            f"{result.crossing_vert_pairs} vertical pair(s)"
        )
    if (
        result.adjacent_horiz_min_spacing is not None
        and result.adjacent_horiz_min_spacing < ADJACENT_SPACING_WARNING_PX
    ):
        result.warnings.append(
            f"tight_spacing: horizontal min spacing {result.adjacent_horiz_min_spacing:.2f}px "
            f"(< {ADJACENT_SPACING_WARNING_PX:.0f}px)"
        )
    if (
        result.adjacent_vert_min_spacing is not None
        and result.adjacent_vert_min_spacing < ADJACENT_SPACING_WARNING_PX
    ):
        result.warnings.append(
            f"tight_spacing: vertical min spacing {result.adjacent_vert_min_spacing:.2f}px "
            f"(< {ADJACENT_SPACING_WARNING_PX:.0f}px)"
        )
    if result.h1_max_clamp is not None and result.h1_max_clamp > H1_CLAMP_WARNING_RATIO:
        result.warnings.append(
            f"high_H1_clamp: {result.h1_max_clamp:.3f} of H1 tail points were clamped "
            f"(> {H1_CLAMP_WARNING_RATIO:.2f})"
        )


def collect_recommendations(result: SpeakerAuditResult) -> list[str]:
    codes = []
    for message in [*result.hard_errors, *result.warnings]:
        code = parse_issue_code(message)
        if code not in codes:
            codes.append(code)
    return codes


def figure_title(speaker: str, status: str, warning_count: int) -> str:
    return f"{speaker}\nstatus: {status} | warnings: {warning_count}"


def save_success_figure(grid: Any, speaker: str, output_path: Path, *, status: str, warning_count: int) -> None:
    fig = visualize_grid(
        grid,
        figsize=(12, 12),
        show_contours=True,
        show_labels=True,
        style=AUDIT_STYLE,
    )
    fig.axes[0].set_title(
        figure_title(speaker, status, warning_count),
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_error_figure(
    speaker: str,
    output_path: Path,
    *,
    image: Any | None,
    contours: dict[str, np.ndarray] | None,
    hard_errors: list[str],
    warnings: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    if image is not None:
        ax.imshow(image, cmap="gray")
        try:
            img_arr = np.asarray(image)
            height, width = img_arr.shape[:2]
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
        except Exception:
            pass
    else:
        ax.set_facecolor("#f5f5f5")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    if contours:
        for name, pts in sorted(contours.items()):
            pts_arr = np.asarray(pts, dtype=float)
            if len(pts_arr) == 0:
                continue
            color = "#ff006e" if name == "tongue" else "#666666"
            lw = 2.0 if name in {"incisior-hard-palate", "mandible-incisior", "pharynx"} else 1.0
            ax.plot(pts_arr[:, 0], pts_arr[:, 1], color=color, linewidth=lw, alpha=0.70)

    details = "\n".join([*hard_errors, *warnings]) if (hard_errors or warnings) else "Unknown build failure"
    ax.text(
        0.02,
        0.98,
        details,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", fc="#a4133c", ec="none", alpha=0.90),
    )
    ax.set_title(figure_title(speaker, "error", len(warnings)), fontsize=13, fontweight="bold")
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_speaker(
    speaker: str,
    *,
    vtnl_dir: Path,
    output_dir: Path,
    n_vert: int,
    n_points: int,
    max_dev_ratio: float,
) -> SpeakerAuditResult:
    output_path = output_dir / "speakers" / f"{speaker}_grid.png"
    image = None
    contours: dict[str, np.ndarray] | None = None

    try:
        image, contours = load_frame_vtnl(speaker, vtnl_dir)
        validation = validate_grid_contours(contours)
        result = SpeakerAuditResult(
            speaker=speaker,
            status="ok",
            hard_errors=list(validation.hard_errors),
            warnings=list(validation.warnings),
            contour_point_counts=dict(validation.point_counts),
            output_image=str(output_path),
        )
        if validation.hard_errors:
            raise GridValidationError("; ".join(validation.hard_errors))

        grid = build_grid(
            image,
            contours,
            n_vert=n_vert,
            n_points=n_points,
            frame_number=0,
            max_dev_ratio=max_dev_ratio,
        )

        result.warnings.extend(message for message in grid.warnings if message not in result.warnings)
        result.landmark_presence = landmark_presence_from_grid(grid)
        all_lines = [*grid.horiz_lines, *grid.vert_lines]
        result.out_of_bounds_points = count_out_of_bounds_points(all_lines, np.asarray(image).shape[:2])
        result.crossing_horiz_pairs = count_crossing_pairs(grid.horiz_lines)
        result.crossing_vert_pairs = count_crossing_pairs(grid.vert_lines)
        result.adjacent_horiz_min_spacing = adjacent_min_spacing(grid.horiz_lines)
        result.adjacent_vert_min_spacing = adjacent_min_spacing(grid.vert_lines)
        result.h1_dev_ratio = float(grid.h1_dev_ratio)
        result.h1_max_clamp = float(grid.h1_max_clamp)
        add_geometry_warnings(result)
        result.recommendations = collect_recommendations(result)
        if result.warnings:
            result.status = "warning"
        save_success_figure(
            grid,
            speaker,
            output_path,
            status=result.status,
            warning_count=result.warning_count,
        )
        return result
    except Exception as exc:
        if contours is not None:
            validation = validate_grid_contours(contours)
            contour_counts = dict(validation.point_counts)
            warnings = list(validation.warnings)
            hard_errors = list(validation.hard_errors)
        else:
            contour_counts = {}
            warnings = []
            hard_errors = []

        if not isinstance(exc, GridValidationError):
            hard_errors.append(f"build_failure: {type(exc).__name__}: {exc}")
        elif not hard_errors:
            hard_errors.append(str(exc))

        result = SpeakerAuditResult(
            speaker=speaker,
            status="error",
            hard_errors=hard_errors,
            warnings=warnings,
            contour_point_counts=contour_counts,
            output_image=str(output_path),
            landmark_presence={key: False for key in [*(f"I{i}" for i in range(1, 8)), "M1", "L6", "P1"]},
        )
        result.recommendations = collect_recommendations(result)
        save_error_figure(
            speaker,
            output_path,
            image=image,
            contours=contours,
            hard_errors=result.hard_errors,
            warnings=result.warnings,
        )
        return result


def severity_key(result: SpeakerAuditResult) -> tuple[float, ...]:
    horiz_penalty = 0.0
    if result.adjacent_horiz_min_spacing is not None:
        horiz_penalty = max(0.0, ADJACENT_SPACING_WARNING_PX - result.adjacent_horiz_min_spacing)
    vert_penalty = 0.0
    if result.adjacent_vert_min_spacing is not None:
        vert_penalty = max(0.0, ADJACENT_SPACING_WARNING_PX - result.adjacent_vert_min_spacing)
    return (
        1.0 if result.status == "error" else 0.0,
        float(result.hard_error_count),
        float(result.warning_count),
        float(result.crossing_horiz_pairs + result.crossing_vert_pairs),
        float(result.out_of_bounds_points),
        float(result.h1_max_clamp or 0.0),
        horiz_penalty,
        vert_penalty,
    )


def save_contact_sheet(
    results: list[SpeakerAuditResult],
    output_path: Path,
    *,
    title: str,
    empty_message: str,
) -> None:
    if not results:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, empty_message, ha="center", va="center", fontsize=14, fontweight="bold")
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    ncols = min(3, max(1, len(results)))
    nrows = int(math.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5.6 * nrows), dpi=150)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax in axes_arr:
        ax.set_axis_off()

    for ax, result in zip(axes_arr, results):
        image = imageio.imread(result.output_image)
        ax.imshow(image)
        ax.set_title(
            f"{result.speaker}\n{result.status} | warn={result.warning_count} | err={result.hard_error_count}",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_summary_payload(args: argparse.Namespace, results: list[SpeakerAuditResult]) -> dict[str, Any]:
    return {
        "workflow": "vtnl_grid_quality_audit",
        "vtnl_dir": str(args.vtnl_dir),
        "output_dir": str(args.output_dir),
        "speaker_count": len(results),
        "ok_count": sum(result.status == "ok" for result in results),
        "warning_count": sum(result.status == "warning" for result in results),
        "error_count": sum(result.status == "error" for result in results),
        "thresholds": {
            "adjacent_spacing_warning_px": ADJACENT_SPACING_WARNING_PX,
            "h1_clamp_warning_ratio": H1_CLAMP_WARNING_RATIO,
            "max_dev_ratio": args.max_dev_ratio,
        },
        "results": [asdict(result) for result in results],
    }


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(path: Path, results: list[SpeakerAuditResult]) -> None:
    contour_names = sorted({name for result in results for name in result.contour_point_counts})
    fieldnames = [
        "speaker",
        "status",
        "hard_errors",
        "warnings",
        "recommendations",
        "out_of_bounds_points",
        "crossing_horiz_pairs",
        "crossing_vert_pairs",
        "adjacent_horiz_min_spacing",
        "adjacent_vert_min_spacing",
        "h1_dev_ratio",
        "h1_max_clamp",
        "has_I1",
        "has_I2",
        "has_I3",
        "has_I4",
        "has_I5",
        "has_I6",
        "has_I7",
        "has_M1",
        "has_L6",
        "has_P1",
        "output_image",
    ] + [f"count_{name}" for name in contour_names]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "speaker": result.speaker,
                "status": result.status,
                "hard_errors": " | ".join(result.hard_errors),
                "warnings": " | ".join(result.warnings),
                "recommendations": " | ".join(result.recommendations),
                "out_of_bounds_points": result.out_of_bounds_points,
                "crossing_horiz_pairs": result.crossing_horiz_pairs,
                "crossing_vert_pairs": result.crossing_vert_pairs,
                "adjacent_horiz_min_spacing": result.adjacent_horiz_min_spacing,
                "adjacent_vert_min_spacing": result.adjacent_vert_min_spacing,
                "h1_dev_ratio": result.h1_dev_ratio,
                "h1_max_clamp": result.h1_max_clamp,
                "has_I1": result.landmark_presence.get("I1", False),
                "has_I2": result.landmark_presence.get("I2", False),
                "has_I3": result.landmark_presence.get("I3", False),
                "has_I4": result.landmark_presence.get("I4", False),
                "has_I5": result.landmark_presence.get("I5", False),
                "has_I6": result.landmark_presence.get("I6", False),
                "has_I7": result.landmark_presence.get("I7", False),
                "has_M1": result.landmark_presence.get("M1", False),
                "has_L6": result.landmark_presence.get("L6", False),
                "has_P1": result.landmark_presence.get("P1", False),
                "output_image": result.output_image,
            }
            for contour_name in contour_names:
                row[f"count_{contour_name}"] = result.contour_point_counts.get(contour_name, 0)
            writer.writerow(row)


def write_summary_md(path: Path, args: argparse.Namespace, results: list[SpeakerAuditResult]) -> None:
    lines = [
        "# VTNL Grid Diagnostics",
        "",
        f"- VTNL dir: `{args.vtnl_dir}`",
        f"- Speakers scanned: {len(results)}",
        f"- OK: {sum(result.status == 'ok' for result in results)}",
        f"- Warning: {sum(result.status == 'warning' for result in results)}",
        f"- Error: {sum(result.status == 'error' for result in results)}",
        "",
        "| speaker | status | hard errors | warnings | out-of-bounds | crossings(H/V) | min H spacing | min V spacing | H1 clamp | recommendations |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]

    for result in results:
        lines.append(
            f"| {result.speaker} | {result.status} | {result.hard_error_count} | {result.warning_count} | "
            f"{result.out_of_bounds_points} | {result.crossing_horiz_pairs}/{result.crossing_vert_pairs} | "
            f"{'-' if result.adjacent_horiz_min_spacing is None else f'{result.adjacent_horiz_min_spacing:.2f}'} | "
            f"{'-' if result.adjacent_vert_min_spacing is None else f'{result.adjacent_vert_min_spacing:.2f}'} | "
            f"{'-' if result.h1_max_clamp is None else f'{result.h1_max_clamp:.3f}'} | "
            f"{', '.join(result.recommendations) if result.recommendations else '-'} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation Guide",
            "",
        ]
    )
    for code, guidance in PATTERN_GUIDANCE.items():
        lines.append(f"- `{code}`: {guidance}")

    issue_results = [result for result in results if result.recommendations]
    if issue_results:
        lines.extend(["", "## Speaker Findings", ""])
        for result in issue_results:
            lines.append(f"### {result.speaker}")
            lines.append(f"- status: `{result.status}`")
            lines.append(f"- recommendations: `{', '.join(result.recommendations)}`")
            if result.hard_errors:
                lines.append(f"- hard errors: `{' | '.join(result.hard_errors)}`")
            if result.warnings:
                lines.append(f"- warnings: `{' | '.join(result.warnings)}`")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "speakers").mkdir(parents=True, exist_ok=True)

    speakers = discover_vtnl_speakers(args.vtnl_dir)
    if not speakers:
        raise FileNotFoundError(f"No VTNL ROI zip files found in {args.vtnl_dir}")

    results = [
        evaluate_speaker(
            speaker,
            vtnl_dir=args.vtnl_dir,
            output_dir=args.output_dir,
            n_vert=args.n_vert,
            n_points=args.n_points,
            max_dev_ratio=args.max_dev_ratio,
        )
        for speaker in speakers
    ]

    ordered_results = sorted(results, key=lambda result: result.speaker)
    issue_results = sorted(
        [result for result in ordered_results if result.is_issue],
        key=severity_key,
        reverse=True,
    )

    payload = build_summary_payload(args, ordered_results)
    write_summary_json(args.output_dir / "summary.json", payload)
    write_summary_csv(args.output_dir / "summary.csv", ordered_results)
    write_summary_md(args.output_dir / "summary.md", args, ordered_results)
    save_contact_sheet(
        ordered_results,
        args.output_dir / "all_speakers_contact_sheet.png",
        title="VTNL Grid Audit: All Speakers",
        empty_message="No speakers found",
    )
    save_contact_sheet(
        issue_results,
        args.output_dir / "issues_contact_sheet.png",
        title="VTNL Grid Audit: Issues Only",
        empty_message="No warnings or hard failures detected",
    )

    print(f"[done] speakers scanned: {len(ordered_results)}")
    print(f"[done] summaries: {args.output_dir}")
    print(f"[done] issues: {len(issue_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
