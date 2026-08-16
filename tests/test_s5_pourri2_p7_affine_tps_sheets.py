from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grid_transform.apps.render_s5_pourri2_p7_affine_tps_sheets import (
    CONTOUR_ORDER,
    REFERENCE_SPEAKER,
    SPEAKERS,
    crop_mri_panel,
    compute_p2cp_rows,
    invert_affine,
    load_pourri2_selection,
    load_review_contours,
    lower_incisor_lateral_controls,
    refine_inverse_mapping,
    square_view_box,
    warp_mri_for_stage,
)
from grid_transform.transform_helpers import apply_transform


def write_selection(path: Path) -> None:
    rows = []
    for index in range(1, 11):
        speaker = f"P{index}"
        rows.append(
            {
                "speaker": speaker,
                "candidate": "pourri/r2",
                "selected_frame": str(700 + index),
                "duration_ms": "100",
                "alignment_source": (
                    "old_textgrid_fallback" if speaker == "P10" else "incoming_*_c"
                ),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_selection_excludes_p2_and_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    write_selection(path)
    rows = load_pourri2_selection(path)
    assert tuple(row["speaker"] for row in rows) == SPEAKERS
    assert "P2" not in {row["speaker"] for row in rows}


def test_load_selection_requires_explicit_p10_fallback(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    write_selection(path)
    text = path.read_text(encoding="utf-8").replace("old_textgrid_fallback", "incoming_*_c")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="P10"):
        load_pourri2_selection(path)


def test_load_review_contours_uses_frozen_case_snapshot(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "P1_S5_F0970"
    contour_dir = case_dir / "contours_npy"
    contour_dir.mkdir(parents=True)
    (case_dir / "metadata.json").write_text(
        '{"speaker":"P1","session":"S5","frame":970}',
        encoding="utf-8",
    )
    expected = np.column_stack(
        [np.linspace(0.0, 135.0, 50), np.linspace(135.0, 0.0, 50)]
    )
    for label in CONTOUR_ORDER:
        np.save(contour_dir / f"{label}.npy", expected)

    contours = load_review_contours(tmp_path, "P1", 970)

    assert tuple(contours) == tuple(sorted(CONTOUR_ORDER))
    np.testing.assert_allclose(contours["tongue"], expected * (480.0 / 136.0))


def test_crop_mri_panel_returns_transform_space_image() -> None:
    frame = np.zeros((600, 450, 3), dtype=np.uint8)
    cropped = crop_mri_panel(frame)
    assert cropped.shape == (480, 480)


def test_inverse_affine_roundtrip_is_exact() -> None:
    affine = {
        "A": np.array([[1.1, 0.2], [-0.1, 0.9]], dtype=float),
        "t": np.array([12.0, -7.0], dtype=float),
    }
    points = np.array([[0.0, 0.0], [40.0, 70.0], [479.0, 479.0]])
    mapped = apply_transform(affine, points)
    recovered = apply_transform(invert_affine(affine), mapped)
    np.testing.assert_allclose(recovered, points, atol=1e-10)


def test_lower_incisor_lateral_controls_are_signed_width_extrema() -> None:
    grid = SimpleNamespace(
        M1=np.array([0.0, 0.0]),
        L6=np.array([0.0, 10.0]),
    )
    contour = np.array(
        [[0.0, 0.0], [3.0, 3.0], [-5.0, 5.0], [2.0, 8.0], [0.0, 10.0]]
    )

    negative, positive = lower_incisor_lateral_controls(contour, grid)

    np.testing.assert_array_equal(negative, np.array([3.0, 3.0]))
    np.testing.assert_array_equal(positive, np.array([-5.0, 5.0]))


def test_stage_mri_warp_uses_inverse_mapping() -> None:
    source = np.zeros((480, 480), dtype=np.uint8)
    source[100, 80] = 255
    translation = np.array([12.0, 7.0])

    def inverse_mapping(points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) - translation

    warped, valid_fraction = warp_mri_for_stage(source, inverse_mapping)
    assert warped[107, 92] == 255
    assert warped[100, 80] == 0
    assert 0.9 < valid_fraction < 1.0


def test_stage_mri_warp_trims_bright_support_border() -> None:
    source = np.full((480, 480), 255, dtype=np.uint8)
    warped, valid_fraction = warp_mri_for_stage(
        source,
        lambda points: np.asarray(points, dtype=float),
        boundary_trim_px=4,
    )
    assert np.all(warped[:4] == 0)
    assert np.all(warped[:, :4] == 0)
    assert warped[240, 240] == 255
    assert 0.95 < valid_fraction < 1.0


def test_refined_inverse_recovers_nonlinear_forward_points() -> None:
    def forward(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return points + 0.001 * points**2

    source = np.array([[20.0, 30.0], [100.0, 180.0], [300.0, 250.0]])
    target = forward(source)
    recovered = refine_inverse_mapping(
        target,
        forward_mapping=forward,
        initial_inverse_mapping=lambda points: np.asarray(points, dtype=float),
    )
    np.testing.assert_allclose(recovered, source, atol=1e-7)


def test_square_view_box_contains_annotations_and_zooms() -> None:
    points = np.array([[100.0, 120.0], [260.0, 390.0]])
    x0, x1, y0, y1 = square_view_box([points], padding=10.0)
    assert x1 - x0 == pytest.approx(y1 - y0)
    assert x1 - x0 < 480.0
    assert x0 <= points[:, 0].min() <= points[:, 0].max() <= x1
    assert y0 <= points[:, 1].min() <= points[:, 1].max() <= y1


def test_identity_p2cp_is_zero_and_reference_is_excluded_from_overall() -> None:
    contour = np.column_stack([np.arange(50, dtype=float), np.zeros(50)])
    ready = {
        speaker: {label: contour.copy() for label in CONTOUR_ORDER} for speaker in SPEAKERS
    }
    stages = {
        "ready": ready,
        "affine": {
            speaker: {label: contour.copy() for label in CONTOUR_ORDER} for speaker in SPEAKERS
        },
        "affine_tps": {
            speaker: {label: contour.copy() for label in CONTOUR_ORDER} for speaker in SPEAKERS
        },
    }

    def mean_metric(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(first - second, axis=1)))

    def rms_metric(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))

    contour_rows, speaker_rows, overall_rows = compute_p2cp_rows(
        stages, mean_metric, rms_metric
    )
    assert len(contour_rows) == 2 * len(SPEAKERS) * len(CONTOUR_ORDER)
    assert len(speaker_rows) == 2 * len(SPEAKERS)
    assert all(row["n_source_speakers"] == len(SPEAKERS) - 1 for row in overall_rows)
    p7_rows = [row for row in speaker_rows if row["speaker"] == REFERENCE_SPEAKER]
    assert all(row["p2cp_rms_contour_macro_mm"] == 0.0 for row in p7_rows)

    _, p10_speaker_rows, p10_overall_rows = compute_p2cp_rows(
        stages,
        mean_metric,
        rms_metric,
        reference_speaker="P10",
    )
    assert {row["reference_speaker"] for row in p10_speaker_rows} == {"P10"}
    assert {row["reference_speaker"] for row in p10_overall_rows} == {"P10"}
    p10_rows = [row for row in p10_speaker_rows if row["speaker"] == "P10"]
    assert all(row["p2cp_rms_contour_macro_mm"] == 0.0 for row in p10_rows)
