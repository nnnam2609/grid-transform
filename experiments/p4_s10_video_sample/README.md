# P4 S10 Video Sample

This folder contains the canonical source files for the `P4/S10` sample-video pipeline, based on the logic in `database_make_video.m` from the ArtSpeech repository.

What it does:

- reads `DCM_2D/S10` and orders frames by DICOM `InstanceNumber`
- reads `OTHER/S10/DENOISED_SOUND_P4_S10.wav`
- parses `TEXT_ALIGNMENT_P4_S10.trs` for sentence spans
- parses `TEXT_ALIGNMENT_P4_S10.textgrid` for word and phonetic spans
- derives frame timing exactly like the MATLAB code:
  `frame_rate = sample_rate / (n_audio_samples / n_dicom_frames)`
- renders a video frame with:
  - MRI image
  - current sentence
  - current word label
  - current phoneme label
  - waveform view with a red time cursor

Run:

```powershell
.\.venv\Scripts\python .\scripts\run\run_build_p4_s10_video.py
```

If your local ArtSpeech root lives somewhere else, pass `--dataset-root <path>`.

Canonical outputs are written to:

- `outputs/videos/p4_s10_video_sample/VIDEO_P4_S10_sample.mp4`
- `outputs/videos/p4_s10_video_sample/annotation_summary.json`
- `outputs/videos/p4_s10_video_sample/annotation_summary.md`
- `outputs/videos/p4_s10_video_sample/word_intervals.csv`
- `outputs/videos/p4_s10_video_sample/phoneme_intervals.csv`
- `outputs/videos/p4_s10_video_sample/sentence_intervals.csv`

You can still run the local script in this folder, but `scripts/run/run_build_p4_s10_video.py` is the preferred entry point.
