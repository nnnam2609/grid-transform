from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from grid_transform.config import PROJECT_DIR, REPORT_OUTPUT_DIR


DATASET_ROOT_DEFAULT = PROJECT_DIR.parent / "Data" / "Artspeech_database"
REPORT_DIR_DEFAULT = REPORT_OUTPUT_DIR / "artspeech_script_report"
SCRIPT_FILE_PATTERN = re.compile(r"TEXT_ALIGNMENT_(P\d+)_([Ss]\d+)\.trs$")
WORD_PATTERN = re.compile(r"[\wÀ-ÿ'-]+", flags=re.UNICODE)
MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â€™", "â€œ", "â€\x9d", "�")
FRENCH_CHARS = "éèêëàâîïôöùûüçœÉÈÊËÀÂÎÏÔÖÙÛÜÇŒ"


@dataclass
class SessionScript:
    speaker: str
    session: str
    session_index: int
    trs_path: Path
    layout_variant: str
    sentences: list[str]

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def full_text(self) -> str:
        return " || ".join(self.sentences)

    @property
    def word_count(self) -> int:
        return sum(len(WORD_PATTERN.findall(sentence)) for sentence in self.sentences)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all ArtSpeech alignment scripts by speaker/session and write a "
            "report with full text plus content notes."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT_DEFAULT,
        help="Root folder of Artspeech_database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR_DEFAULT,
        help="Directory where the report files will be written.",
    )
    return parser.parse_args(argv)


def speaker_sort_key(speaker: str) -> tuple[int, str]:
    return (int(speaker[1:]), speaker)


def session_sort_key(session: str) -> tuple[int, str]:
    return (int(session[1:]), session)


def read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    candidates: list[str] = []
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            decoded = raw.decode(encoding)
            candidates.append(repair_mojibake(decoded))
        except UnicodeDecodeError:
            continue
    if candidates:
        return min(candidates, key=text_quality_key)
    return repair_mojibake(raw.decode("utf-8", errors="replace"))


def repair_mojibake(text: str) -> str:
    candidates = [text]
    for encoding in ("latin-1", "cp1252"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return min(candidates, key=text_quality_key)


def text_quality_key(text: str) -> tuple[int, int]:
    mojibake_score = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    french_bonus = sum(text.count(char) for char in FRENCH_CHARS)
    replacement_penalty = text.count("�")
    return (mojibake_score + replacement_penalty, -french_bonus)


def parse_trs_sentences(path: Path) -> list[str]:
    text = read_text_best_effort(path)
    matches = list(re.finditer(r'<Sync time="([^"]+)"\s*/>', text))
    sentences: list[str] = []
    for current, nxt in zip(matches, matches[1:]):
        segment = text[current.end() : nxt.start()]
        segment = re.sub(r"<[^>]+>", " ", segment)
        sentence = " ".join(segment.split())
        if sentence:
            sentences.append(sentence)
    return sentences


def detect_layout_variant(dataset_root: Path, trs_path: Path) -> str:
    parts = trs_path.relative_to(dataset_root).parts
    if len(parts) >= 4 and parts[1] == "OTHER":
        return "direct"
    if len(parts) >= 5 and parts[2] == "OTHER":
        return "nested"
    return "other"


def discover_session_scripts(dataset_root: Path) -> list[SessionScript]:
    records: list[SessionScript] = []
    for trs_path in sorted(dataset_root.rglob("TEXT_ALIGNMENT_*.trs")):
        match = SCRIPT_FILE_PATTERN.fullmatch(trs_path.name)
        if not match:
            continue
        speaker = match.group(1)
        session = match.group(2).upper()
        records.append(
            SessionScript(
                speaker=speaker,
                session=session,
                session_index=int(session[1:]),
                trs_path=trs_path,
                layout_variant=detect_layout_variant(dataset_root, trs_path),
                sentences=parse_trs_sentences(trs_path),
            )
        )
    records.sort(key=lambda row: (speaker_sort_key(row.speaker), session_sort_key(row.session)))
    return records


def sentence_word_count(sentence: str) -> int:
    return len(WORD_PATTERN.findall(sentence))


def variant_example(
    majority_record: SessionScript,
    other_record: SessionScript,
) -> dict[str, str | int]:
    differing_index = 0
    for index, (majority_sentence, other_sentence) in enumerate(
        zip(majority_record.sentences, other_record.sentences),
        start=1,
    ):
        if majority_sentence != other_sentence:
            differing_index = index
            break
    else:
        if len(majority_record.sentences) != len(other_record.sentences):
            differing_index = min(len(majority_record.sentences), len(other_record.sentences)) + 1
        else:
            differing_index = 1

    majority_sentence = (
        majority_record.sentences[differing_index - 1]
        if differing_index - 1 < len(majority_record.sentences)
        else "[missing]"
    )
    other_sentence = (
        other_record.sentences[differing_index - 1]
        if differing_index - 1 < len(other_record.sentences)
        else "[missing]"
    )
    return {
        "session": majority_record.session,
        "majority_speaker": majority_record.speaker,
        "variant_speaker": other_record.speaker,
        "sentence_index": differing_index,
        "majority_sentence": majority_sentence,
        "variant_sentence": other_sentence,
    }


def build_summary(records: list[SessionScript], dataset_root: Path) -> dict[str, object]:
    by_speaker: dict[str, list[SessionScript]] = defaultdict(list)
    by_session_name: dict[str, list[SessionScript]] = defaultdict(list)
    sentence_counter: Counter[str] = Counter()
    sentence_lengths: list[int] = []
    session_sentence_hist: Counter[int] = Counter()
    layout_counter: Counter[str] = Counter()

    for record in records:
        by_speaker[record.speaker].append(record)
        by_session_name[record.session].append(record)
        session_sentence_hist[record.sentence_count] += 1
        layout_counter[record.layout_variant] += 1
        for sentence in record.sentences:
            sentence_counter[sentence] += 1
            sentence_lengths.append(sentence_word_count(sentence))

    speaker_rows = []
    for speaker in sorted(by_speaker, key=speaker_sort_key):
        speaker_records = by_speaker[speaker]
        speaker_rows.append(
            {
                "speaker": speaker,
                "session_count": len(speaker_records),
                "sentence_count": sum(row.sentence_count for row in speaker_records),
                "layout_variants": sorted({row.layout_variant for row in speaker_records}),
                "first_trs_path": str(speaker_records[0].trs_path),
            }
        )

    session_variant_rows = []
    discrepancy_examples: list[dict[str, str | int]] = []
    for session in sorted(by_session_name, key=session_sort_key):
        session_records = by_session_name[session]
        signature_groups: dict[str, list[SessionScript]] = defaultdict(list)
        for record in session_records:
            signature_groups[record.full_text].append(record)
        grouped = sorted(signature_groups.values(), key=lambda rows: (-len(rows), rows[0].speaker))
        majority_group = grouped[0]
        majority_record = majority_group[0]
        variant_count = len(grouped)
        session_variant_rows.append(
            {
                "session": session,
                "variant_count": variant_count,
                "majority_group_size": len(majority_group),
                "majority_speakers": [row.speaker for row in majority_group],
                "majority_script": majority_record.full_text,
            }
        )
        if variant_count > 1:
            for candidate_group in grouped[1:]:
                candidate = candidate_group[0]
                if candidate.full_text != majority_record.full_text:
                    discrepancy_examples.append(variant_example(majority_record, candidate))
                    break

    discrepancy_examples.sort(key=lambda row: session_sort_key(str(row["session"])))

    stable_sessions = sorted(
        session_variant_rows,
        key=lambda row: (row["variant_count"], -row["majority_group_size"], session_sort_key(str(row["session"]))),
    )
    unstable_sessions = sorted(
        session_variant_rows,
        key=lambda row: (-row["variant_count"], row["majority_group_size"], session_sort_key(str(row["session"]))),
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "speaker_count": len(by_speaker),
        "session_doc_count": len(records),
        "sentence_count": sum(record.sentence_count for record in records),
        "unique_sentence_count": len(sentence_counter),
        "avg_words_per_sentence": (sum(sentence_lengths) / len(sentence_lengths)) if sentence_lengths else 0.0,
        "min_words_per_sentence": min(sentence_lengths) if sentence_lengths else 0,
        "max_words_per_sentence": max(sentence_lengths) if sentence_lengths else 0,
        "sentences_per_session_hist": dict(sorted(session_sentence_hist.items())),
        "layout_variant_counts": dict(sorted(layout_counter.items())),
        "speakers": speaker_rows,
        "session_variants": session_variant_rows,
        "stable_sessions": stable_sessions[:5],
        "unstable_sessions": unstable_sessions[:5],
        "most_common_sentences": [
            {"sentence": sentence, "count": count}
            for sentence, count in sentence_counter.most_common(15)
        ],
        "discrepancy_examples": discrepancy_examples[:8],
        "singleton_sentence_count": sum(1 for count in sentence_counter.values() if count == 1),
        "shared_sentence_count_ge_5": sum(1 for count in sentence_counter.values() if count >= 5),
    }


def format_sentence_block(sentences: list[str]) -> str:
    return "<br>".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1))


def write_session_csv(records: list[SessionScript], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "speaker",
                "session",
                "layout_variant",
                "sentence_count",
                "word_count",
                "trs_path",
                "full_text",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.speaker,
                    record.session,
                    record.layout_variant,
                    record.sentence_count,
                    record.word_count,
                    str(record.trs_path),
                    record.full_text,
                ]
            )


def write_sentence_csv(records: list[SessionScript], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "speaker",
                "session",
                "sentence_index",
                "word_count",
                "sentence",
                "trs_path",
            ]
        )
        for record in records:
            for index, sentence in enumerate(record.sentences, start=1):
                writer.writerow(
                    [
                        record.speaker,
                        record.session,
                        index,
                        sentence_word_count(sentence),
                        sentence,
                        str(record.trs_path),
                    ]
                )


def write_summary_json(summary: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(records: list[SessionScript], summary: dict[str, object], output_path: Path) -> None:
    by_speaker: dict[str, list[SessionScript]] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker].append(record)

    lines = [
        "# ArtSpeech Script Report",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Dataset root: `{summary['dataset_root']}`",
        "",
        "## Coverage",
        "",
        f"- Speakers: `{summary['speaker_count']}`",
        f"- Session documents: `{summary['session_doc_count']}`",
        f"- Sentence intervals: `{summary['sentence_count']}`",
        f"- Unique sentence strings: `{summary['unique_sentence_count']}`",
        f"- Avg words per sentence: `{summary['avg_words_per_sentence']:.2f}`",
        f"- Word-count range per sentence: `{summary['min_words_per_sentence']}` to `{summary['max_words_per_sentence']}`",
        f"- Session length histogram: `{summary['sentences_per_session_hist']}`",
        f"- Layout variants found: `{summary['layout_variant_counts']}`",
        "",
        "## Content Evaluation",
        "",
        "- The corpus is a French read-speech prompt set built from short sentences and articulator-heavy phrases.",
        "- Content is often semantically odd but syntactically valid, which is consistent with phonetically rich recording prompts rather than natural discourse.",
        "- The scripts mix long descriptive sentences with very short repeated prompts such as `Trois sacs carrés`, `Pour tout casser.`, `Très acariâtre.`, `Nous palissons.`, and `Des abat-jour.`",
        "- Session labels are only partially standardized across speakers: many sessions reuse the same core prompt set, but transcripts still show speaker-specific typos, insertions, omissions, and substitutions.",
        f"- There are `{summary['singleton_sentence_count']}` sentence strings that appear only once and `{summary['shared_sentence_count_ge_5']}` sentence strings that appear at least five times.",
        "",
        "Most stable session labels by transcript agreement:",
        "",
        "| Session | Distinct variants | Majority size |",
        "| --- | ---: | ---: |",
    ]

    for row in summary["stable_sessions"]:
        lines.append(
            f"| {row['session']} | {row['variant_count']} | {row['majority_group_size']} |"
        )

    lines.extend(
        [
            "",
            "Most variable session labels by transcript agreement:",
            "",
            "| Session | Distinct variants | Majority size |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in summary["unstable_sessions"]:
        lines.append(
            f"| {row['session']} | {row['variant_count']} | {row['majority_group_size']} |"
        )

    lines.extend(
        [
            "",
            "Most frequent sentence strings:",
            "",
            "| Count | Sentence |",
            "| ---: | --- |",
        ]
    )
    for row in summary["most_common_sentences"]:
        lines.append(f"| {row['count']} | {row['sentence']} |")

    lines.extend(
        [
            "",
            "Representative transcript discrepancies:",
            "",
            "| Session | Majority | Variant | Sentence # | Majority text | Variant text |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in summary["discrepancy_examples"]:
        lines.append(
            "| {session} | {majority_speaker} | {variant_speaker} | {sentence_index} | {majority_sentence} | {variant_sentence} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Speaker Coverage",
            "",
            "| Speaker | Sessions | Sentences | Layout | Example path |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["speakers"]:
        lines.append(
            f"| {row['speaker']} | {row['session_count']} | {row['sentence_count']} | "
            f"{', '.join(row['layout_variants'])} | {row['first_trs_path']} |"
        )

    lines.append("")
    lines.append("## All Scripts By Speaker And Session")
    lines.append("")
    for speaker in sorted(by_speaker, key=speaker_sort_key):
        lines.append(f"### {speaker}")
        lines.append("")
        lines.append("| Session | # sentences | Text |")
        lines.append("| --- | ---: | --- |")
        for record in sorted(by_speaker[speaker], key=lambda row: session_sort_key(row.session)):
            lines.append(
                f"| {record.session} | {record.sentence_count} | {format_sentence_block(record.sentences)} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_session_scripts(dataset_root)
    if not records:
        raise FileNotFoundError(f"No TEXT_ALIGNMENT_*.trs files were found under {dataset_root}")

    summary = build_summary(records, dataset_root)

    markdown_path = output_dir / "artspeech_script_report.md"
    session_csv_path = output_dir / "speaker_session_scripts.csv"
    sentence_csv_path = output_dir / "speaker_session_sentences.csv"
    summary_json_path = output_dir / "summary.json"

    write_markdown(records, summary, markdown_path)
    write_session_csv(records, session_csv_path)
    write_sentence_csv(records, sentence_csv_path)
    write_summary_json(summary, summary_json_path)

    print(
        json.dumps(
            {
                "markdown_report": str(markdown_path),
                "session_csv": str(session_csv_path),
                "sentence_csv": str(sentence_csv_path),
                "summary_json": str(summary_json_path),
                "session_docs": summary["session_doc_count"],
                "sentences": summary["sentence_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
