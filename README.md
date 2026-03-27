# Grid Transform

Research utilities for building vocal-tract grids, aligning speakers with `Affine + TPS`, and testing transfer hypotheses across VTNL references, segmentation targets, and ArtSpeech sessions.

The repo is organized for reproducible experiments rather than a packaged library. Reusable logic lives in `grid_transform/`, and the canonical command surface lives in `scripts/run/`.

## What This Repo Covers

- Build a landmark-driven grid for a VTNL reference or segmentation target.
- Estimate a two-step `Affine + TPS` transform between a source speaker and a target frame.
- Move articulators or warp the full source image into target space.
- Generate report figures and a PDF summary.
- Project VTNL annotations onto ArtSpeech videos when per-frame ArtSpeech contours are unavailable.
- Interactively edit a projected source annotation on an ArtSpeech frame, save it, and reuse it for fixed-annotation session warping.
- Warp a full ArtSpeech session into a target frame under a fixed-session annotation assumption.
- Inspect within-speaker and cross-speaker vowel variability from ArtSpeech sessions.

## Repo Layout

- `grid_transform/`
  Shared Python modules, path config, notebook bootstrap, and app entrypoints.
- `scripts/run/`
  Official wrappers for the main workflows.
- `notebooks/`
  Exploration notebooks. Notebook outputs are intentionally not treated as versioned results.
- `experiments/`
  Report generation and side experiment code.
- `docs/`
  Public-facing documentation, data notes, and README assets.
- `VTNL/`
  Lightweight bundled VTNL reference images and ROI zips used by the default examples.
- `vocal-tract-seg/`
  Lightweight bundled target-frame sample data used by the default examples.

Root scripts such as `create_speaker_grid.py` remain available as compatibility shims, but `scripts/run/` is the primary interface.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Data Expectations

The base grid-transform examples run against the bundled sample/reference data in `VTNL/` and `vocal-tract-seg/`.

ArtSpeech session utilities require a separate external dataset root. The expected structure is described in [`docs/DATA.md`](docs/DATA.md). In short, the code expects an ArtSpeech-style speaker layout with:

- `DCM_2D/<session>/` for frame-level DICOM data
- `OTHER/<session>/` for `wav`, `textgrid`, and `trs` session metadata

Large external datasets are not redistributed in this repository.

## Quick Start

### Core Grid Transform Pipeline

```powershell
.\.venv\Scripts\python .\scripts\run\run_create_speaker_grid.py --source vtnl --speaker 1640_s10_0829
.\.venv\Scripts\python .\scripts\run\run_method4_transform.py
.\.venv\Scripts\python .\scripts\run\run_move_target_articulators.py
.\.venv\Scripts\python .\scripts\run\run_warp_source_speaker_to_target.py
```

### Report Generation

```powershell
.\.venv\Scripts\python .\scripts\run\run_generate_report_assets.py
.\.venv\Scripts\python .\scripts\run\run_convert_to_pdf.py
.\scripts\run\build_report.ps1
```

### ArtSpeech Session Utilities

```powershell
.\.venv\Scripts\python .\scripts\run\run_build_artspeech_session_video.py --speaker P7 --session S10 --dataset-root <ARTSPEECH_ROOT>
.\.venv\Scripts\python .\scripts\run\run_project_vtnl_reference_to_artspeech_video.py --target-speaker 1640_s10_0829 --artspeech-speaker P7 --session S10 --dataset-root <ARTSPEECH_ROOT>
.\.venv\Scripts\python .\scripts\run\run_warp_artspeech_session_to_target_video.py --annotation-speaker 1640_s10_0829 --artspeech-speaker P7 --session S10 --target-frame 143020 --dataset-root <ARTSPEECH_ROOT> --output-mode both
```

### Interactive Source Annotation Editor

The source-annotation editor opens a local `cv2` window, projects a VTNL reference contour set onto the best-matching ArtSpeech frame, lets you drag contour handles, then saves the edited annotation for reuse in the session warp pipeline.

Default example for the current bundled workflow:

```powershell
.\.venv\Scripts\python .\scripts\run\run_edit_source_annotation.py --artspeech-speaker P7 --session S2 --reference-speaker 1640_s10_0829 --target-frame 143020 --dataset-root <ARTSPEECH_ROOT>
```

Interactive controls:

- Drag handles in the top-left source panel to adjust the projected annotation.
- Press `s` to save the edited annotation and preview images only.
- Press `v` to save the edited annotation and immediately render the warped session outputs.
- Press `r` to reset back to the initial projected annotation.
- Press `q` or `Esc` to close the editor.

Default save location:

```text
outputs/source_annotation_edits/p7_s2_frame_0829/
```

Saved files:

- `edited_annotation.json`: canonical contour coordinates plus source/target metadata.
- `source_full_head_annotation.png`: source frame with the editable contour overlay.
- `source_grid_annotation.png`: source grid rebuilt from the current annotation.
- `source_to_target_preview.png`: fixed-frame warped preview in target space.
- `editor_overview.png`: 4-panel snapshot matching the editor layout.
- `save_summary.json`: summary of the saved assets and any triggered sequence render.
- `sequence_warp/`: created when rendering is requested; contains warped/review videos, preview PNGs, and `warp_summary.json`.

Useful options:

- `--source-frame <N>`: override the auto-selected best-match ArtSpeech frame.
- `--output-dir <PATH>`: write the saved annotation bundle somewhere other than the default `outputs/source_annotation_edits/...` folder.
- `--output-mode both|warped|review`: choose which session-render outputs to create when `v` is used.
- `--max-output-frames <N>`: render only the first `N` frames for a quick smoke test.
- `--no-gui`: run the same initialization/save flow without opening the interactive window.
- `--skip-video-on-save`: only meaningful with `--no-gui`; save the annotation bundle without rendering the session warp.

Headless save-only example:

```powershell
.\.venv\Scripts\python .\scripts\run\run_edit_source_annotation.py --artspeech-speaker P7 --session S2 --reference-speaker 1640_s10_0829 --target-frame 143020 --dataset-root <ARTSPEECH_ROOT> --no-gui --skip-video-on-save
```

Reuse a saved edited annotation in the warp pipeline:

```powershell
.\.venv\Scripts\python .\scripts\run\run_warp_artspeech_session_to_target_video.py --artspeech-speaker P7 --session S2 --target-frame 143020 --source-annotation-json outputs\source_annotation_edits\p7_s2_frame_0829\edited_annotation.json --dataset-root <ARTSPEECH_ROOT> --output-mode both
```

Current tracked example from the saved `P7/S2` annotation bundle:

- Source frame: `829`
- Source time: `16.5533s`
- Reference VTNL frame: `1640_s10_0829`
- Target frame: `143020`
- Match correlation before manual edit: `0.9866`

Editor overview and saved annotation outputs:

<p>
  <img src="docs/assets/github/p7-s2-editor-overview.png" alt="P7 S2 editor overview" width="49%" />
  <img src="docs/assets/github/p7-s2-source-full-head-annotation.png" alt="P7 S2 source full head annotation" width="49%" />
</p>
<p>
  <img src="docs/assets/github/p7-s2-source-grid-annotation.png" alt="P7 S2 source grid annotation" width="49%" />
  <img src="docs/assets/github/p7-s2-source-to-target-preview.png" alt="P7 S2 source to target preview" width="49%" />
</p>

Sequence warp preview from the saved annotation:

![P7 S2 sequence review preview](docs/assets/github/p7-s2-sequence-review-preview.png)

Review video: [P7_S2_warped_to_143020_review.mp4](docs/assets/github/p7-s2-warped-to-143020-review.mp4)

### Vowel Variability Utilities

```powershell
.\.venv\Scripts\python .\scripts\run\run_extract_all_speakers_vowel_variants.py --dataset-root <ARTSPEECH_ROOT> --samples-per-vowel 10
.\.venv\Scripts\python .\scripts\run\run_extract_speaker_vowel_variants.py --speaker P7 --dataset-root <ARTSPEECH_ROOT> --samples-per-vowel 10
.\.venv\Scripts\python .\scripts\run\run_analyze_aligned_speaker_vowel_variants.py --speaker P7 --samples-per-vowel 10
```

## Results Gallery

### Default Target Grid

![Target grid](docs/assets/github/target-grid.png)

### Default Source Grid

![Source grid](docs/assets/github/source-grid.png)

### Two-Step `Affine + TPS` Alignment

![Two-step alignment](docs/assets/github/two-step-alignment.png)

### Articulators Moved Into Target Space

![Moved articulators](docs/assets/github/moved-articulators.png)

### Full Source Image Warped Into Target Space

![Warped source comparison](docs/assets/github/warped-source-comparison.png)

### Same-Speaker Vowel Variants

![P7 vowel contact sheet](docs/assets/github/p7-a-contact-sheet.png)

### Same-Speaker Vowel Prototype vs Outliers After Alignment

![P7 aligned vowel ranking](docs/assets/github/p7-a-prototype-vs-outliers.png)

## Reproducibility Notes

- Generated outputs live under `outputs/` and are treated as disposable rerun artifacts.
- `docs/assets/github/` contains the curated static images used by this README; it is the only tracked results-like asset folder.
- The ArtSpeech projection workflow uses a fixed image-space resize from a VTNL reference; it is not a contour-derived per-frame warp.
- The full-session ArtSpeech warp also assumes a fixed-session geometry for the source session; it does not estimate new ArtSpeech contours frame by frame.
- Notebooks are exploratory and intentionally kept secondary to the reusable Python modules and wrapper scripts.

## Limitations

- The repo currently assumes specific landmark conventions such as `I1..I7`, `C1..C6`, `M1`, `L6`, and `P1`.
- Several workflows are designed around the bundled example pair: VTNL `1640_s10_0829` and segmentation frame `143020`.
- ArtSpeech analyses rely on external aligned labels and on fixed-session assumptions when full contour annotations are unavailable.

## Release Checklist

See [`docs/release_checklist.md`](docs/release_checklist.md) for a short pre-publish checklist.

## License

This repository is released under the [`MIT License`](LICENSE).
