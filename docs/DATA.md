# DATA

This repository mixes three data tiers:

1. lightweight bundled reference data in `VTLN/`
2. lightweight bundled target-frame sample data in `VTLN/data/`
3. larger external ArtSpeech session data used by the video and vowel-analysis utilities

The goal of this note is to document the **path contract** expected by the code, not to prescribe any machine-specific mount or download workflow.

## What Is Bundled In This Repo

### `VTLN/`

The repo bundles a canonical `VTLN/data/` folder used by the default examples.

Expected file pattern:

- `data/<image_name>.png` or `data/<image_name>.tif`
- `data/<image_name>.zip`
- `data/nnunet_data_80/<case>/test/PNG_MR/<frame>.png`
- `data/nnunet_data_80/<case>/test/contours/<frame>.zip`
- `data/nnunet_data_80/<case>/test/contours/<frame>_*.roi`

The zip contains ImageJ ROI contours keyed by articulator name.

Examples:

- `data/1640_P7_S2_F0829.png`
- `data/1640_P7_S2_F0829.zip`
- `data/nnunet_data_80/2008-003^01-1791/test/PNG_MR/143020.png`
- `data/nnunet_data_80/2008-003^01-1791/test/contours/143020.zip`

The bundled default example uses target frame `143020` from case `2008-003^01-1791/test`.

## Canonical Annotation Update Flow

Interactive annotation edits are saved under `outputs/`, not written directly into `VTLN/data/`.

Current pipeline behavior:

- `outputs/annotation_to_grid_transform/.../source_annotation.latest.json` stores the latest source annotation saved from the multi-step cv2 workflow.
- `outputs/source_annotation_edits/.../edited_annotation.json` stores the latest source annotation saved from the standalone source-annotation editor.
- For a given `(artspeech_speaker, session, source_frame)`, the newest saved JSON from those locations is treated as the authoritative annotation update.
- `scripts/run/run_sync_latest_annotations_to_curated_vtln.py` applies that update back onto the canonical curated VTLN selection in place by overwriting the matching `*.png` and `*.zip` files after archiving the previous versions.
- `scripts/run/run_build_vtln_data_bundle.py` rebuilds the public `VTLN/data/` bundle from the triplet manifest and those newest saved annotations. If no newer saved annotation exists for a row, the builder falls back to the curated ZIP already present in the curated VTLN folder.

In other words, the saved annotation snapshot becomes the new default/canonical annotation only when it is promoted through the sync/build maintenance commands.

## External ArtSpeech Dataset Contract

The session-oriented utilities expect an ArtSpeech-style dataset root supplied with `--dataset-root`.

Minimum expected layout:

```text
<ARTSPEECH_ROOT>/
  <speaker>/
    <speaker>/
      DCM_2D/
        <session>/
          *.dcm
      OTHER/
        <session>/
          DENOISED_SOUND_<speaker>_<session>.wav
          TEXT_ALIGNMENT_<speaker>_<session>.textgrid
          TEXT_ALIGNMENT_<speaker>_<session>.trs
```

Typical examples of speaker/session ids are:

- speaker: `P7`
- session: `S10`

The ArtSpeech workflows in this repo do not require the full dataset to be copied into the repository. They only require that the external dataset root follows the layout above.

## Supported File Types

### VTLN reference loading

- RGB triplet or grayscale image: `.png`, `.tif`, or `.tiff`
- contours: `.zip` containing ImageJ `.roi` files

### Segmentation target loading

- frame images: `PNG_MR/<frame>.png`
- groundtruth contours: `contours/<frame>.zip` or `contours/<frame>_*.roi`

### ArtSpeech session loading

- MRI session frames: `*.dcm`
- audio: `.wav`
- Praat alignment: `.textgrid`
- sentence transcript timing: `.trs`

## Label and Annotation Notes

The ArtSpeech utilities assume a Praat `TextGrid` with at least:

- a word-like tier
- a phoneme-like tier

The code also consumes `.trs` transcripts for sentence-level timing when building review videos.

Phone labels are expected to follow the same style as the token inventory used by the underlying ArtSpeech resources, including symbols such as:

- `a`, `e`, `i`, `o`, `u`
- `E`, `O`, `R`
- stop closures such as `t_cl` or `p_cl`
- silence/unlabeled spans such as `#` or blank intervals

## Which Workflows Need Which Data

- Core grid transform pipeline:
  works with the bundled `VTLN/data/` examples by default.
- Report generation:
  uses the same bundled sample/reference data unless you override paths in the scripts.
- ArtSpeech session video, projection, warp, and vowel-variability workflows:
  require an external ArtSpeech dataset root.

## Redistribution Note

This repository is intended to distribute:

- code
- notebooks
- lightweight example/reference assets

It is **not** intended to redistribute large external ArtSpeech datasets. For public sharing, point collaborators to the dataset provider or to your lab's approved access route instead of bundling the full corpus into the repo.
