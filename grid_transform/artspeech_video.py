from __future__ import annotations

import csv
import json
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pydicom

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.config import PROJECT_DIR, VIDEO_OUTPUT_DIR


DEFAULT_SPEAKER = "P4"
DEFAULT_SESSION = "S10"


@dataclass
class Interval:
    start: float
    end: float
    text: str


@dataclass
class Tier:
    name: str
    intervals: list[Interval]


@dataclass
class SessionPaths:
    dataset_root: Path
    speaker: str
    session: str
    dcm_dir: Path
    other_dir: Path
    wav_path: Path
    textgrid_path: Path
    trs_path: Path


@dataclass
class SessionData:
    paths: SessionPaths
    images: np.ndarray
    dicom_metadata: list[dict[str, object]]
    samples: np.ndarray
    sample_rate: int
    tiers: list[Tier]
    sentences: list[Interval]
    frame_rate: float
    frame_min: float
    frame_max: float


@dataclass
class LabelSnapshot:
    interval_index: int
    frame_index: int
    time_s: float
    word: str
    phoneme: str
    sentence: str


class IntervalCursor:
    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = intervals
        self.index = 0

    def at(self, time_s: float) -> str:
        if not self.intervals:
            return ""
        while self.index + 1 < len(self.intervals) and time_s > self.intervals[self.index].end:
            self.index += 1
        while self.index > 0 and time_s < self.intervals[self.index].start:
            self.index -= 1
        current = self.intervals[self.index]
        if current.start <= time_s <= current.end:
            return current.text.strip()
        return ""


def dataset_root_candidates(speaker: str) -> tuple[Path, ...]:
    data_root = PROJECT_DIR.parent / "Data"
    return (
        data_root / speaker / speaker,
        data_root / "Artspeech_database" / speaker / speaker,
    )


def resolve_default_dataset_root(speaker: str = DEFAULT_SPEAKER) -> Path:
    candidates = dataset_root_candidates(speaker)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_output_dir(speaker: str, session: str) -> Path:
    return VIDEO_OUTPUT_DIR / f"{speaker.lower()}_{session.lower()}"


def video_filename(speaker: str, session: str) -> str:
    return f"VIDEO_{speaker}_{session}_sample.mp4"


def build_session_paths(dataset_root: Path, speaker: str, session: str) -> SessionPaths:
    dcm_dir = dataset_root / "DCM_2D" / session
    other_dir = dataset_root / "OTHER" / session
    wav_path = other_dir / f"DENOISED_SOUND_{speaker}_{session}.wav"
    textgrid_path = other_dir / f"TEXT_ALIGNMENT_{speaker}_{session}.textgrid"
    trs_path = other_dir / f"TEXT_ALIGNMENT_{speaker}_{session}.trs"
    return SessionPaths(
        dataset_root=dataset_root,
        speaker=speaker,
        session=session,
        dcm_dir=dcm_dir,
        other_dir=other_dir,
        wav_path=wav_path,
        textgrid_path=textgrid_path,
        trs_path=trs_path,
    )


def ensure_inputs_exist(paths: SessionPaths) -> None:
    required = [
        paths.dcm_dir,
        paths.other_dir,
        paths.wav_path,
        paths.textgrid_path,
        paths.trs_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input paths:\n" + "\n".join(missing))


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_textgrid(path: Path) -> list[Tier]:
    lines = read_text(path).splitlines()
    tiers: list[Tier] = []
    i = 0
    item_pattern = re.compile(r"item \[(\d+)\]:")
    while i < len(lines):
        line = lines[i].strip()
        if item_pattern.fullmatch(line):
            name = ""
            intervals: list[Interval] = []
            i += 1
            while i < len(lines):
                current = lines[i].strip()
                if item_pattern.fullmatch(current):
                    i -= 1
                    break
                if current.startswith("name ="):
                    name = current.split("=", 1)[1].strip().strip('"')
                elif current.startswith("intervals ["):
                    start = float(lines[i + 1].split("=", 1)[1].strip())
                    end = float(lines[i + 2].split("=", 1)[1].strip())
                    label = lines[i + 3].split("=", 1)[1].strip().strip('"')
                    intervals.append(Interval(start=start, end=end, text=label))
                    i += 3
                i += 1
            tiers.append(Tier(name=name, intervals=intervals))
        i += 1
    return tiers


def parse_trs(path: Path) -> list[Interval]:
    text = read_text(path)
    matches = list(re.finditer(r'<Sync time="([^"]+)"\s*/>', text))
    sentences: list[Interval] = []
    for current, nxt in zip(matches, matches[1:]):
        start = float(current.group(1))
        end = float(nxt.group(1))
        raw_segment = text[current.end() : nxt.start()]
        segment = re.sub(r"<[^>]+>", " ", raw_segment)
        sentence = " ".join(segment.split())
        if sentence:
            sentences.append(Interval(start=start, end=end, text=sentence))
    return sentences


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


def load_dicom_series(dcm_dir: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    files = sorted(path for path in dcm_dir.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dcm_dir}")

    frames_by_index: dict[int, np.ndarray] = {}
    metadata_by_index: dict[int, dict[str, object]] = {}
    max_index = 0

    for path in files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=False)
        index = int(ds.InstanceNumber)
        max_index = max(max_index, index)
        frames_by_index[index] = ds.pixel_array
        metadata_by_index[index] = {
            "source_file": path.name,
            "instance_number": index,
            "rows": int(ds.Rows),
            "columns": int(ds.Columns),
            "sequence_name": str(ds.get("SequenceName", "")),
            "pixel_spacing": [float(value) for value in ds.get("PixelSpacing", [])],
            "repetition_time": float(ds.get("RepetitionTime", 0.0) or 0.0),
            "echo_time": float(ds.get("EchoTime", 0.0) or 0.0),
        }

    first = next(iter(frames_by_index.values()))
    stack = np.zeros((max_index, first.shape[0], first.shape[1]), dtype=first.dtype)
    metadata: list[dict[str, object]] = []
    for index in range(1, max_index + 1):
        frame = frames_by_index.get(index)
        if frame is None:
            continue
        stack[index - 1] = frame
        metadata.append(metadata_by_index[index])

    if np.all(stack[0] == 0):
        nonzero = np.flatnonzero(stack.reshape(stack.shape[0], -1).sum(axis=1))
        if nonzero.size:
            stack = stack[nonzero[0] :]
            metadata = metadata[nonzero[0] :]
    return stack, metadata


def compute_frame_rate(samples: np.ndarray, sample_rate: int, n_frames: int) -> float:
    step = samples.size / n_frames
    return sample_rate / step


def load_session_data(dataset_root: Path | str, speaker: str, session: str) -> SessionData:
    dataset_root = Path(dataset_root)
    paths = build_session_paths(dataset_root, speaker, session)
    ensure_inputs_exist(paths)

    images, dicom_metadata = load_dicom_series(paths.dcm_dir)
    samples, sample_rate = read_wav_mono(paths.wav_path)
    tiers = parse_textgrid(paths.textgrid_path)
    sentences = parse_trs(paths.trs_path)
    frame_rate = compute_frame_rate(samples, sample_rate, images.shape[0])
    return SessionData(
        paths=paths,
        images=images,
        dicom_metadata=dicom_metadata,
        samples=samples,
        sample_rate=sample_rate,
        tiers=tiers,
        sentences=sentences,
        frame_rate=frame_rate,
        frame_min=float(images.min()),
        frame_max=float(images.max()),
    )


def downsample_waveform(samples: np.ndarray, sample_rate: int, max_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    if samples.size <= max_points:
        step = 1
    else:
        step = math.ceil(samples.size / max_points)
    indices = np.arange(0, samples.size, step)
    return indices / sample_rate, samples[indices]


def classify_tier(tier: Tier) -> str:
    labels = [interval.text.strip() for interval in tier.intervals if interval.text.strip() and interval.text.strip() != "#"]
    if not labels:
        return "empty"
    average_length = sum(len(label) for label in labels) / len(labels)
    if average_length <= 2.2 and len(labels) > 100:
        return "phonetic-like"
    return "word-like"


def normalize_frame(frame: np.ndarray, frame_min: float, frame_max: float) -> np.ndarray:
    if frame_max <= frame_min:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = (frame.astype(np.float32) - frame_min) / (frame_max - frame_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def normalized_session_frame(session_data: SessionData, frame_index: int) -> np.ndarray:
    return normalize_frame(session_data.images[frame_index], session_data.frame_min, session_data.frame_max)


def write_interval_csv(path: Path, intervals: list[Interval]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["start_sec", "end_sec", "label"])
        for interval in intervals:
            writer.writerow([f"{interval.start:.6f}", f"{interval.end:.6f}", interval.text])


def write_summary(
    output_dir: Path,
    session_data: SessionData,
) -> None:
    tier_summaries = []
    for index, tier in enumerate(session_data.tiers, start=1):
        labels = [interval.text.strip() for interval in tier.intervals if interval.text.strip()]
        unique_labels = list(dict.fromkeys(labels))
        tier_summaries.append(
            {
                "tier_index": index,
                "tier_name": tier.name,
                "tier_kind_guess": classify_tier(tier),
                "interval_count": len(tier.intervals),
                "sample_labels": unique_labels[:20],
            }
        )

    summary = {
        "speaker": session_data.paths.speaker,
        "session": session_data.paths.session,
        "dataset_root": str(session_data.paths.dataset_root),
        "dcm_dir": str(session_data.paths.dcm_dir),
        "wav_path": str(session_data.paths.wav_path),
        "textgrid_path": str(session_data.paths.textgrid_path),
        "trs_path": str(session_data.paths.trs_path),
        "dicom_frame_count": len(session_data.dicom_metadata),
        "audio_sample_rate": session_data.sample_rate,
        "audio_sample_count": int(session_data.samples.size),
        "audio_duration_sec": session_data.samples.size / session_data.sample_rate,
        "derived_frame_rate_fps": session_data.frame_rate,
        "dicom_example_metadata": {
            "first": session_data.dicom_metadata[0],
            "middle": session_data.dicom_metadata[len(session_data.dicom_metadata) // 2],
            "last": session_data.dicom_metadata[-1],
        },
        "textgrid_tiers": tier_summaries,
        "sentence_count": len(session_data.sentences),
        "sentences": [
            {"start_sec": interval.start, "end_sec": interval.end, "text": interval.text}
            for interval in session_data.sentences
        ],
        "notes": [
            "Video timing follows the MATLAB script: frame_rate = sample_rate / (n_audio_samples / n_dicom_frames).",
            "Frame timestamps are aligned to frame centers: (index + 0.5) / frame_rate.",
            "Tier kind is inferred heuristically from interval density and label length.",
        ],
    }

    (output_dir / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_lines = [
        f"Speaker/session: {session_data.paths.speaker}/{session_data.paths.session}",
        f"DICOM frames: {len(session_data.dicom_metadata)}",
        f"Audio: {session_data.sample_rate} Hz, {session_data.samples.size / session_data.sample_rate:.6f} s",
        f"Derived frame rate: {session_data.frame_rate:.6f} fps",
        "",
        "TextGrid tiers:",
    ]
    for tier in tier_summaries:
        summary_lines.append(
            f"- Tier {tier['tier_index']}: {tier['tier_name']} | {tier['tier_kind_guess']} | "
            f"{tier['interval_count']} intervals | sample labels: {', '.join(tier['sample_labels'][:10])}"
        )
    summary_lines.extend(["", "Sentence intervals from TRS:"])
    for interval in session_data.sentences:
        summary_lines.append(f"- {interval.start:.3f}-{interval.end:.3f}s: {interval.text}")
    (output_dir / "annotation_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def make_figure(
    first_frame: np.ndarray,
    waveform_time: np.ndarray,
    waveform_values: np.ndarray,
    session_label: str,
) -> tuple[plt.Figure, dict[str, object]]:
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.12)

    ax_image = fig.add_subplot(grid[0])
    ax_wave = fig.add_subplot(grid[1])

    ax_image.set_axis_off()
    image_artist = ax_image.imshow(first_frame, cmap="gray", vmin=0, vmax=255)
    session_text = ax_image.text(
        0.02,
        0.97,
        session_label,
        color="yellow",
        fontsize=12,
        va="top",
        transform=ax_image.transAxes,
    )
    frame_text = ax_image.text(
        0.02,
        0.03,
        "I1",
        color="yellow",
        fontsize=12,
        va="bottom",
        transform=ax_image.transAxes,
    )
    word_text = ax_image.text(
        0.50,
        0.03,
        "",
        color="yellow",
        fontsize=13,
        ha="center",
        va="bottom",
        transform=ax_image.transAxes,
    )
    phoneme_text = ax_image.text(
        0.50,
        0.97,
        "",
        color="yellow",
        fontsize=13,
        ha="center",
        va="top",
        transform=ax_image.transAxes,
    )

    ax_wave.plot(waveform_time, waveform_values, color="black", linewidth=0.8)
    time_cursor = ax_wave.axvline(0.0, color="red", linewidth=1.2)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_ylabel("Amp.")
    ax_wave.set_xlabel("Time (s)")
    title = ax_wave.set_title("", fontsize=9)

    artists = {
        "image": image_artist,
        "session_text": session_text,
        "frame_text": frame_text,
        "word_text": word_text,
        "phoneme_text": phoneme_text,
        "time_cursor": time_cursor,
        "title": title,
        "ax_wave": ax_wave,
    }
    return fig, artists


def draw_frame(
    fig: plt.Figure,
    artists: dict[str, object],
    frame: np.ndarray,
    frame_index: int,
    current_time: float,
    audio_duration: float,
    sentence: str,
    word: str,
    phoneme: str,
) -> np.ndarray:
    artists["image"].set_data(frame)
    artists["frame_text"].set_text(f"I{frame_index}")
    artists["word_text"].set_text(word)
    artists["phoneme_text"].set_text(phoneme)
    artists["time_cursor"].set_xdata([current_time, current_time])
    artists["title"].set_text(sentence)

    window = 1.0
    xmin = max(0.0, current_time - window)
    xmax = min(audio_duration, current_time + window)
    if xmax - xmin < 0.2:
        xmax = min(audio_duration, xmin + 0.2)
    artists["ax_wave"].set_xlim(xmin, xmax)

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write_video(
    output_dir: Path,
    session_data: SessionData,
    max_frames: int = 0,
) -> Path:
    frame_count = session_data.images.shape[0] if max_frames <= 0 else min(max_frames, session_data.images.shape[0])
    frame_times = (np.arange(frame_count, dtype=np.float64) + 0.5) / session_data.frame_rate
    audio_duration = session_data.samples.size / session_data.sample_rate

    word_tier = session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else []
    phoneme_tier = session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else []
    word_cursor = IntervalCursor(word_tier)
    phoneme_cursor = IntervalCursor(phoneme_tier)
    sentence_cursor = IntervalCursor(session_data.sentences)

    waveform_time, waveform_values = downsample_waveform(session_data.samples, session_data.sample_rate)
    first_frame = normalized_session_frame(session_data, 0)
    fig, artists = make_figure(first_frame, waveform_time, waveform_values, session_data.paths.session)

    output_video = output_dir / video_filename(session_data.paths.speaker, session_data.paths.session)
    preview_indices = {0: output_dir / "preview_frame_0001.png"}
    preview_indices[frame_count // 2] = output_dir / f"preview_frame_{frame_count // 2 + 1:04d}.png"

    with imageio.get_writer(
        output_video,
        fps=session_data.frame_rate,
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        ffmpeg_log_level="warning",
        audio_path=str(session_data.paths.wav_path),
        audio_codec="aac",
    ) as writer:
        for index in range(frame_count):
            if index % 100 == 0 or index == frame_count - 1:
                print(f"[video] rendering frame {index + 1}/{frame_count}")
            current_time = frame_times[index]
            frame = normalized_session_frame(session_data, index)
            sentence = sentence_cursor.at(current_time)
            word = word_cursor.at(current_time)
            phoneme = phoneme_cursor.at(current_time)
            rgb = draw_frame(
                fig=fig,
                artists=artists,
                frame=frame,
                frame_index=index + 1,
                current_time=current_time,
                audio_duration=audio_duration,
                sentence=sentence,
                word=word,
                phoneme=phoneme,
            )
            writer.append_data(rgb)
            preview_path = preview_indices.get(index)
            if preview_path is not None and not preview_path.exists():
                imageio.imwrite(preview_path, rgb)
    plt.close(fig)
    return output_video


def filter_labeled_intervals(intervals: list[Interval]) -> list[Interval]:
    return [interval for interval in intervals if interval.text.strip() and interval.text.strip() != "#"]


def frame_index_for_time(time_s: float, frame_rate: float, frame_count: int) -> int:
    raw_index = int(round(time_s * frame_rate - 0.5))
    return int(np.clip(raw_index, 0, frame_count - 1))


def evenly_spaced_indices(length: int, max_items: int) -> list[int]:
    if length <= max_items:
        return list(range(length))
    indices = np.linspace(0, length - 1, num=max_items)
    unique_indices: list[int] = []
    for value in indices:
        idx = int(round(float(value)))
        if idx not in unique_indices:
            unique_indices.append(idx)
    return unique_indices


def build_label_snapshots(session_data: SessionData, max_samples: int = 6) -> list[LabelSnapshot]:
    word_intervals = filter_labeled_intervals(session_data.tiers[0].intervals if session_data.tiers else [])
    if not word_intervals:
        return []

    word_cursor = IntervalCursor(session_data.tiers[0].intervals if session_data.tiers else [])
    phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) > 1 else [])
    sentence_cursor = IntervalCursor(session_data.sentences)

    snapshots: list[LabelSnapshot] = []
    for interval_idx in evenly_spaced_indices(len(word_intervals), max_samples):
        interval = word_intervals[interval_idx]
        time_s = 0.5 * (interval.start + interval.end)
        frame_index = frame_index_for_time(time_s, session_data.frame_rate, session_data.images.shape[0])
        snapshots.append(
            LabelSnapshot(
                interval_index=interval_idx,
                frame_index=frame_index,
                time_s=time_s,
                word=word_cursor.at(time_s),
                phoneme=phoneme_cursor.at(time_s),
                sentence=sentence_cursor.at(time_s),
            )
        )
    return snapshots


def run_session_video(
    speaker: str,
    session: str,
    *,
    dataset_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    max_frames: int = 0,
) -> dict[str, str]:
    dataset_root = Path(dataset_root) if dataset_root is not None else resolve_default_dataset_root(speaker)
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir(speaker, session)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] reading DICOM series")
    paths = build_session_paths(dataset_root, speaker, session)
    ensure_inputs_exist(paths)
    images, dicom_metadata = load_dicom_series(paths.dcm_dir)
    print("[load] reading audio")
    samples, sample_rate = read_wav_mono(paths.wav_path)
    print("[load] parsing annotations")
    tiers = parse_textgrid(paths.textgrid_path)
    sentences = parse_trs(paths.trs_path)
    session_data = SessionData(
        paths=paths,
        images=images,
        dicom_metadata=dicom_metadata,
        samples=samples,
        sample_rate=sample_rate,
        tiers=tiers,
        sentences=sentences,
        frame_rate=compute_frame_rate(samples, sample_rate, images.shape[0]),
        frame_min=float(images.min()),
        frame_max=float(images.max()),
    )

    write_interval_csv(output_dir / "word_intervals.csv", session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
    write_interval_csv(output_dir / "phoneme_intervals.csv", session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
    write_interval_csv(output_dir / "sentence_intervals.csv", session_data.sentences)
    write_summary(output_dir, session_data)

    print("[video] writing MP4")
    video_path = write_video(output_dir=output_dir, session_data=session_data, max_frames=max_frames)

    result = {
        "video_path": str(video_path),
        "annotation_summary_json": str(output_dir / "annotation_summary.json"),
        "annotation_summary_md": str(output_dir / "annotation_summary.md"),
        "word_intervals_csv": str(output_dir / "word_intervals.csv"),
        "phoneme_intervals_csv": str(output_dir / "phoneme_intervals.csv"),
        "sentence_intervals_csv": str(output_dir / "sentence_intervals.csv"),
    }
    (output_dir / "run_outputs.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result
