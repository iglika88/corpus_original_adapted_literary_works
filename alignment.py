#!/usr/bin/env python3
"""Align full literary texts with their adapted versions.

The script divides each adapted text into sentence-complete excerpts and aligns
them monotonically with sentence spans from the corresponding full text. It
uses multilingual sentence embeddings and iteratively adjusts adjacent
boundaries when both affected alignment scores improve.

Requirements
------------
Install the dependencies in your Python environment:

    pip install numpy pandas nltk sentence-transformers

Input
-----
Create a UTF-8 CSV manifest with one text pair per row and these columns:

    full_path,adapted_path,language,level,book

``full_path``, ``adapted_path``, and ``language`` are required. ``level`` and
``book`` are optional metadata fields. Paths may be absolute or relative to the
manifest. Supported language codes are en, fr, ru, es/sp, it, and ja/jp.

Example:

    full_path,adapted_path,language,level,book
    texts/work_full.txt,texts/work_adapted.txt,en,B1,work

Usage
-----

    python alignment.py pairs.csv results

Run ``python alignment.py --help`` to see all configurable parameters. The
output directory receives combined CSV files and, unless disabled, equivalent
per-language files.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_COLUMNS = [
    "id",
    "full_text",
    "adapted_text",
    "language",
    "level",
    "book",
    "base_book",
    "similarity",
    "full_units",
    "adapted_units",
    "full_to_adapted_ratio",
]

LANGUAGE_ALIASES = {"sp": "es", "jp": "ja"}
NLTK_LANGUAGES = {
    "en": "english",
    "fr": "french",
    "ru": "russian",
    "es": "spanish",
    "it": "italian",
}
JAPANESE_END = re.compile(r"([。！？!?])")


@dataclass(frozen=True)
class Settings:
    model_name: str
    similarity_threshold: float
    max_ratio: float
    max_full_words: int
    max_full_ja_chars: int
    target_words: int
    min_words: int
    max_words: int
    target_ja_chars: int
    min_ja_chars: int
    max_ja_chars: int
    max_passes: int
    min_improvement: float
    split_parts: int
    split_if_excerpts_gt: int


@dataclass(frozen=True)
class Alignment:
    original_sentences: list[str]
    adapted_chunks: list[str]
    boundaries: list[int]
    similarities: list[float]


def normalize_language(code: str) -> str:
    code = str(code).strip().lower()
    return LANGUAGE_ALIASES.get(code, code)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def split_japanese_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        buffer = ""
        for part in JAPANESE_END.split(line):
            if not part:
                continue
            buffer += part
            if JAPANESE_END.fullmatch(part):
                if buffer.strip():
                    sentences.append(buffer.strip())
                buffer = ""
        if buffer.strip():
            sentences.append(buffer.strip())
    return sentences


def split_sentences(text: str, language: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if language == "ja":
        return [s for s in split_japanese_sentences(text) if normalize_whitespace(s)]

    nltk_language = NLTK_LANGUAGES.get(language)
    if nltk_language is None:
        raise ValueError(f"Unsupported language code: {language!r}")
    try:
        sentences = sent_tokenize(text, language=nltk_language)
    except LookupError:
        # NLTK 3.8+ may require both resources, depending on the environment.
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = sent_tokenize(text, language=nltk_language)
    return [normalize_whitespace(s) for s in sentences if normalize_whitespace(s)]


def unit_count(text: str, language: str) -> int:
    """Count words, or non-whitespace characters for Japanese."""
    if language == "ja":
        return len(re.sub(r"\s+", "", text))
    return len(re.findall(r"\S+", normalize_whitespace(text)))


def adapted_chunk_limits(language: str, settings: Settings) -> tuple[int, int, int]:
    if language == "ja":
        return settings.target_ja_chars, settings.min_ja_chars, settings.max_ja_chars
    return settings.target_words, settings.min_words, settings.max_words


def full_chunk_limit(language: str, settings: Settings) -> int:
    return settings.max_full_ja_chars if language == "ja" else settings.max_full_words


def chunk_sentences_near_target(
    sentences: Sequence[str],
    language: str,
    target_units: int,
    min_units: int,
    max_units: int,
) -> list[str]:
    """Build sentence-complete chunks within soft size limits."""
    if not (0 < min_units <= target_units <= max_units):
        raise ValueError("Chunk sizes must satisfy 0 < min <= target <= max.")

    chunks: list[str] = []
    current: list[str] = []
    current_units = 0

    for sentence in sentences:
        sentence_units = unit_count(sentence, language)
        if not current:
            current = [sentence]
            current_units = sentence_units
            continue

        if current_units + sentence_units <= max_units:
            current.append(sentence)
            current_units += sentence_units
        elif current_units >= min_units:
            chunks.append(" ".join(current))
            current = [sentence]
            current_units = sentence_units
        else:
            # Permit a modest overflow rather than creating a very short chunk.
            current.append(sentence)
            chunks.append(" ".join(current))
            current = []
            current_units = 0

    if current:
        chunks.append(" ".join(current))
    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def build_adapted_chunks(
    sentences: Sequence[str], language: str, settings: Settings
) -> list[str]:
    target, minimum, maximum = adapted_chunk_limits(language, settings)
    return chunk_sentences_near_target(
        sentences, language, target, minimum, maximum
    )


def equal_sentence_boundaries(sentence_count: int, chunk_count: int) -> list[int]:
    if chunk_count <= 0 or chunk_count > sentence_count:
        raise ValueError("Chunk count must be between 1 and the number of sentences.")
    boundaries = [0]
    boundaries.extend(
        int(round(k * sentence_count / chunk_count)) for k in range(1, chunk_count)
    )
    boundaries.append(sentence_count)
    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1]:
            boundaries[index] = boundaries[index - 1] + 1
    boundaries[-1] = sentence_count
    return boundaries


def chunk_full_text_under_limit(
    sentences: Sequence[str], language: str, limit: int
) -> list[int]:
    """Return sentence boundaries whose spans do not normally exceed limit."""
    boundaries = [0]
    current_units = 0
    index = 0
    while index < len(sentences):
        sentence_units = unit_count(sentences[index], language)
        if current_units == 0 and sentence_units > limit:
            # A single overlong sentence must remain intact.
            boundaries.append(index + 1)
            index += 1
        elif current_units + sentence_units <= limit or current_units == 0:
            current_units += sentence_units
            index += 1
        else:
            boundaries.append(index)
            current_units = 0
    if boundaries[-1] != len(sentences):
        boundaries.append(len(sentences))
    return boundaries


def chunk_exactly_by_sentence_count(
    sentences: Sequence[str], chunk_count: int
) -> list[str]:
    boundaries = equal_sentence_boundaries(len(sentences), chunk_count)
    return [
        normalize_whitespace(" ".join(sentences[boundaries[k] : boundaries[k + 1]]))
        for k in range(chunk_count)
    ]


def align_fixed_chunk_count(
    original_sentences: list[str],
    adapted_chunks: list[str],
    model: SentenceTransformer,
    settings: Settings,
) -> Alignment:
    """Align N adapted chunks to N monotonic original-text sentence spans."""
    chunk_count = len(adapted_chunks)
    sentence_count = len(original_sentences)
    boundaries = equal_sentence_boundaries(sentence_count, chunk_count)

    original_embeddings = model.encode(
        original_sentences,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    adapted_embeddings = model.encode(
        adapted_chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    cumulative = np.zeros(
        (original_embeddings.shape[0] + 1, original_embeddings.shape[1]),
        dtype=np.float32,
    )
    cumulative[1:] = np.cumsum(original_embeddings, axis=0)

    def span_mean(start: int, end: int) -> np.ndarray:
        vector = (cumulative[end] - cumulative[start]) / max(end - start, 1)
        norm = np.linalg.norm(vector)
        return vector / (norm if norm > 1e-12 else 1.0)

    def similarity(k: int, current: list[int]) -> float:
        start, end = current[k], current[k + 1]
        return float(span_mean(start, end) @ adapted_embeddings[k])

    def all_similarities(current: list[int]) -> list[float]:
        return [similarity(k, current) for k in range(chunk_count)]

    def average_similarity(current: list[int]) -> float:
        return float(np.mean(all_similarities(current)))

    def refinement_pass(current: list[int], direction: str) -> list[int]:
        refined = current.copy()
        indices = (
            range(chunk_count - 1)
            if direction == "forward"
            else range(chunk_count - 2, -1, -1)
        )
        for k in indices:
            improved = True
            while improved:
                improved = False
                left_score = similarity(k, refined)
                right_score = similarity(k + 1, refined)
                best_boundary: int | None = None
                best_gain = 0.0

                for candidate in (refined[k + 1] + 1, refined[k + 1] - 1):
                    left_size = candidate - refined[k]
                    right_size = refined[k + 2] - candidate
                    if left_size < 1 or right_size < 1:
                        continue
                    previous = refined[k + 1]
                    refined[k + 1] = candidate
                    new_left = similarity(k, refined)
                    new_right = similarity(k + 1, refined)
                    refined[k + 1] = previous

                    if new_left > left_score + 1e-8 and new_right > right_score + 1e-8:
                        gain = (new_left - left_score) + (new_right - right_score)
                        if gain > best_gain:
                            best_gain = gain
                            best_boundary = candidate

                if best_boundary is not None:
                    refined[k + 1] = best_boundary
                    improved = True
        return refined

    current = boundaries.copy()
    current_average = average_similarity(current)
    best_boundaries = current.copy()
    best_average = current_average

    for pass_number in range(1, settings.max_passes + 1):
        direction = "forward" if pass_number % 2 else "backward"
        candidate = refinement_pass(current, direction)
        candidate_average = average_similarity(candidate)
        if candidate_average <= current_average + settings.min_improvement:
            break
        current = candidate
        current_average = candidate_average
        if candidate_average > best_average + settings.min_improvement:
            best_boundaries = candidate.copy()
            best_average = candidate_average

    return Alignment(
        original_sentences=original_sentences,
        adapted_chunks=adapted_chunks,
        boundaries=best_boundaries,
        similarities=[float(score) for score in all_similarities(best_boundaries)],
    )


def align_pair(
    original_text: str,
    adapted_text: str,
    language: str,
    model: SentenceTransformer,
    settings: Settings,
) -> tuple[Alignment | None, str | None]:
    original_sentences = split_sentences(original_text, language)
    adapted_sentences = split_sentences(adapted_text, language)
    if not original_sentences or not adapted_sentences:
        return None, "No sentences remained after sentence splitting."

    adapted_chunks = build_adapted_chunks(adapted_sentences, language, settings)
    if not adapted_chunks:
        return None, "Adapted-text chunking produced no excerpts."

    limit = full_chunk_limit(language, settings)
    if len(adapted_chunks) <= len(original_sentences):
        approximate = equal_sentence_boundaries(
            len(original_sentences), len(adapted_chunks)
        )
        approximate_sizes = [
            unit_count(
                " ".join(original_sentences[approximate[k] : approximate[k + 1]]),
                language,
            )
            for k in range(len(adapted_chunks))
        ]
        if max(approximate_sizes) <= limit:
            return (
                align_fixed_chunk_count(
                    original_sentences, adapted_chunks, model, settings
                ),
                None,
            )

    # Increase N when necessary to keep original-text spans within the limit.
    full_boundaries = chunk_full_text_under_limit(
        original_sentences, language, limit
    )
    required_chunks = len(full_boundaries) - 1
    if required_chunks > len(adapted_sentences):
        return None, (
            f"The full text requires {required_chunks} chunks to stay under "
            f"the {limit}-unit limit, but the adapted text has only "
            f"{len(adapted_sentences)} sentences."
        )
    adapted_chunks = chunk_exactly_by_sentence_count(
        adapted_sentences, required_chunks
    )
    return (
        align_fixed_chunk_count(original_sentences, adapted_chunks, model, settings),
        None,
    )


def split_sentences_into_parts(
    sentences: Sequence[str], part_count: int
) -> list[str]:
    boundaries = equal_sentence_boundaries(len(sentences), part_count)
    return [
        normalize_whitespace(" ".join(sentences[boundaries[k] : boundaries[k + 1]]))
        for k in range(part_count)
    ]


def prepare_parts(
    original_text: str,
    adapted_text: str,
    language: str,
    base_book: str,
    settings: Settings,
) -> list[tuple[str, str, str]]:
    """Split long pairs into corresponding coarse parts before alignment."""
    adapted_sentences = split_sentences(adapted_text, language)
    prospective_chunks = build_adapted_chunks(
        adapted_sentences, language, settings
    ) if adapted_sentences else []

    if (
        settings.split_parts <= 1
        or len(prospective_chunks) <= settings.split_if_excerpts_gt
    ):
        return [(original_text, adapted_text, base_book)]

    original_sentences = split_sentences(original_text, language)
    if (
        len(original_sentences) < settings.split_parts
        or len(adapted_sentences) < settings.split_parts
    ):
        return [(original_text, adapted_text, base_book)]

    original_parts = split_sentences_into_parts(
        original_sentences, settings.split_parts
    )
    adapted_parts = split_sentences_into_parts(
        adapted_sentences, settings.split_parts
    )
    return [
        (original_parts[i], adapted_parts[i], f"{base_book}_part{i + 1}")
        for i in range(settings.split_parts)
    ]


def resolve_input_path(value: object, manifest_directory: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else manifest_directory / path


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"full_path", "adapted_path", "language"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    for optional in ("level", "book"):
        if optional not in manifest.columns:
            manifest[optional] = ""
    if manifest.empty:
        raise ValueError("Manifest contains no text pairs.")
    return manifest


def write_outputs(
    rows: list[dict[str, object]],
    output_directory: Path,
    similarity_threshold: float,
    max_ratio: float,
    save_raw: bool,
    per_language: bool,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    filtered = raw[raw["similarity"] >= similarity_threshold].copy()
    ratio_filtered = filtered[
        filtered["full_to_adapted_ratio"] <= max_ratio
    ].copy()

    for frame in (filtered, ratio_filtered):
        frame.reset_index(drop=True, inplace=True)
        frame["id"] = np.arange(1, len(frame) + 1)

    datasets: list[tuple[str, pd.DataFrame]] = [
        ("aligned_excerpts", filtered),
        ("aligned_excerpts_ratio_filtered", ratio_filtered),
    ]
    if save_raw:
        datasets.append(("aligned_excerpts_raw", raw))

    for name, frame in datasets:
        frame.to_csv(output_directory / f"{name}.csv", index=False, encoding="utf-8")
        if per_language and not frame.empty:
            for language in sorted(frame["language"].unique()):
                frame[frame["language"] == language].to_csv(
                    output_directory / f"{name}_{language}.csv",
                    index=False,
                    encoding="utf-8",
                )

    print(f"Raw aligned excerpts: {len(raw)}")
    print(f"Similarity-filtered excerpts: {len(filtered)}")
    print(f"Similarity- and ratio-filtered excerpts: {len(ratio_filtered)}")
    print(f"Output directory: {output_directory.resolve()}")


def run(args: argparse.Namespace) -> None:
    settings = Settings(
        model_name=args.model,
        similarity_threshold=args.similarity_threshold,
        max_ratio=args.max_ratio,
        max_full_words=args.max_full_words,
        max_full_ja_chars=args.max_full_ja_chars,
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        target_ja_chars=args.target_ja_chars,
        min_ja_chars=args.min_ja_chars,
        max_ja_chars=args.max_ja_chars,
        max_passes=args.max_passes,
        min_improvement=args.min_improvement,
        split_parts=args.split_parts,
        split_if_excerpts_gt=args.split_if_excerpts_gt,
    )
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    model = SentenceTransformer(settings.model_name)

    output_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    next_id = 1

    for row_number, pair in manifest.iterrows():
        language = normalize_language(pair["language"])
        full_path = resolve_input_path(pair["full_path"], manifest_path.parent)
        adapted_path = resolve_input_path(pair["adapted_path"], manifest_path.parent)
        level = str(pair.get("level", "")).strip()
        base_book = str(pair.get("book", "")).strip() or full_path.stem

        if not full_path.is_file() or not adapted_path.is_file():
            missing = [str(p) for p in (full_path, adapted_path) if not p.is_file()]
            skipped.append(
                f"Row {row_number + 2} ({base_book}): missing file(s): {', '.join(missing)}"
            )
            continue

        original_text = read_text(full_path)
        adapted_text = read_text(adapted_path)
        parts = prepare_parts(
            original_text, adapted_text, language, base_book, settings
        )

        for part_original, part_adapted, part_book in parts:
            alignment, reason = align_pair(
                part_original, part_adapted, language, model, settings
            )
            if alignment is None:
                skipped.append(f"Row {row_number + 2} ({part_book}): {reason}")
                continue

            for k, score in enumerate(alignment.similarities):
                start = alignment.boundaries[k]
                end = alignment.boundaries[k + 1]
                full_chunk = normalize_whitespace(
                    " ".join(alignment.original_sentences[start:end])
                )
                adapted_chunk = alignment.adapted_chunks[k]
                full_units = unit_count(full_chunk, language)
                adapted_units = max(unit_count(adapted_chunk, language), 1)
                output_rows.append(
                    {
                        "id": next_id,
                        "full_text": full_chunk,
                        "adapted_text": adapted_chunk,
                        "language": language,
                        "level": level,
                        "book": part_book,
                        "base_book": base_book,
                        "similarity": score,
                        "full_units": full_units,
                        "adapted_units": adapted_units,
                        "full_to_adapted_ratio": full_units / adapted_units,
                    }
                )
                next_id += 1

    write_outputs(
        output_rows,
        args.output_directory,
        settings.similarity_threshold,
        settings.max_ratio,
        args.save_raw,
        not args.no_per_language,
    )
    if skipped:
        print("\nSkipped inputs:")
        for message in skipped:
            print(f"- {message}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monotonically align full texts with adapted versions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="UTF-8 CSV describing text pairs")
    parser.add_argument("output_directory", type=Path, help="directory for output CSV files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    parser.add_argument("--similarity-threshold", type=float, default=0.35)
    parser.add_argument("--max-ratio", type=float, default=3.0)
    parser.add_argument("--max-full-words", type=positive_int, default=4000)
    parser.add_argument("--max-full-ja-chars", type=positive_int, default=9000)
    parser.add_argument("--target-words", type=positive_int, default=300)
    parser.add_argument("--min-words", type=positive_int, default=220)
    parser.add_argument("--max-words", type=positive_int, default=380)
    parser.add_argument("--target-ja-chars", type=positive_int, default=220)
    parser.add_argument("--min-ja-chars", type=positive_int, default=160)
    parser.add_argument("--max-ja-chars", type=positive_int, default=300)
    parser.add_argument("--max-passes", type=positive_int, default=12)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--split-parts", type=positive_int, default=3)
    parser.add_argument("--split-if-excerpts-gt", type=positive_int, default=6)
    parser.add_argument("--save-raw", action="store_true", help="also save unfiltered alignments")
    parser.add_argument("--no-per-language", action="store_true", help="omit per-language CSV files")
    return parser


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
