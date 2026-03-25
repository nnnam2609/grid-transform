# DATA

This repository mixes three data tiers:

1. lightweight bundled reference data in `VTNL/`
2. lightweight bundled target-frame sample data in `vocal-tract-seg/`
3. larger external ArtSpeech session data used by the video and vowel-analysis utilities

The goal of this note is to document the **path contract** expected by the code, not to prescribe any machine-specific mount or download workflow.

## What Is Bundled In This Repo

### `VTNL/`

The repo bundles a small set of VTNL-style reference images and ROI zips used by the default examples.

Expected file pattern:

- `<image_name>.png` or `<image_name>.tif`
- `<image_name>.zip`

The zip contains ImageJ ROI contours keyed by articulator name.

Examples:

- `1640_s10_0654.png`
- `1640_s10_0654.zip`

### `vocal-tract-seg/`

The repo also bundles a lightweight segmentation example used by the default source/target pipeline.

Relevant path pattern:

- `vocal-tract-seg/data_80/<case>/test/PNG_MR/<frame>.png`
- `vocal-tract-seg/results/nnunet_080/inference_contours/<case>/test/<frame>_*.npy`

The bundled default example uses target frame `143020` from case `2008-003^01-1791/test`.

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

### VTNL reference loading

- grayscale image: `.png`, `.tif`, or `.tiff`
- contours: `.zip` containing ImageJ `.roi` files

### Segmentation target loading

- frame images: `PNG_MR/<frame>.png`
- predicted contour arrays: `<frame>_*.npy`

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
  works with the bundled `VTNL/` and `vocal-tract-seg/` examples by default.
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
