#!/usr/bin/env python3
"""Preprocess literary texts and optionally OCR Japanese PDFs without furigana.

The script provides three commands:

1. Clean one UTF-8 text file::

       python preprocess.py text INPUT.txt OUTPUT.txt --language fr

2. Clean a batch described by a UTF-8 CSV manifest::

       python preprocess.py batch manifest.csv OUTPUT_DIRECTORY

   Required manifest columns: ``input_path`` and ``language``.
   Optional columns: ``output_name`` and ``is_play``. Relative input paths are
   resolved from the manifest's directory. ``is_play`` accepts true/false,
   yes/no, or 1/0; play mode preserves short dialogue and heading-like lines.

3. Extract horizontal Japanese text from a PDF while filtering likely
   furigana (ruby) tokens::

       python preprocess.py japanese-pdf INPUT.pdf OUTPUT_DIRECTORY

Text cleaning uses only the Python standard library. Batch mode additionally
requires pandas. Japanese PDF extraction requires pandas, numpy, PyMuPDF,
OpenCV, pytesseract, a Tesseract executable, and Japanese Tesseract language
data. Install the Python packages with::

    pip install pandas numpy pymupdf opencv-python pytesseract

Run ``python preprocess.py COMMAND --help`` for configurable parameters. The
program never overwrites an existing output unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LANGUAGE_ALIASES = {"sp": "es", "jp": "ja"}
PUNCTUATION_END = set(".!?…»”\"'؛;:)]。！？」』】")
HIRAGANA_TOKEN = re.compile(r"^[\u3041-\u3096\u309D\u309E\u30FC]+$")
JAPANESE_CHAR = re.compile(r"[\u3041-\u30FF\u3400-\u9FFF]")

# Chapter and front-matter labels observed in the source collection.
CHAPTER_KEYWORDS = {"CHAPITRE", "PROLOGUE", "PISTE", "TOME", "LIVRE", "PARTIE"}
STARTING_LABELS = {
    "capitolo", "acto", "escena", "chapter", "letter", "book", "volume",
    "part", "stave", "parte", "libro", "traduzione", "trduzione",
}
SCANNER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*Scanned with CS CamScanner\s*$",
        r"^\s*CS CamScanner Scanned with\s*$",
        r"^\s*CS\s*$",
        r"^\s*Scanned with CamScanner\s*$",
        r"^\s*Scanned with\s*$",
        r"^\s*CS CamScanner\s*$",
    )
]


@dataclass(frozen=True)
class TextOptions:
    language: str
    is_play: bool = False
    remove_websites: bool = True
    remove_bracketed_text: bool = True
    remove_headings: bool = True
    remove_glossary_lines: bool = True
    remove_ocr_fragments: bool = True


def normalize_language(code: str) -> str:
    code = str(code).strip().lower()
    return LANGUAGE_ALIASES.get(code, code)


def strip_trailing_spaces(text: str) -> str:
    return re.sub(r"[ \t]+$", "", text)


def ends_with_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in PUNCTUATION_END


def word_tokens(text: str) -> list[str]:
    """Return script-agnostic word-like tokens for line heuristics."""
    tokens: list[str] = []
    current: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith(("L", "N")) or character in "'’-":
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def letters_only(text: str) -> str:
    return "".join(character for character in text if character.isalpha())


def is_all_caps(text: str) -> bool:
    letters = letters_only(text)
    return bool(letters) and letters == letters.upper()


def starts_with_uppercase(text: str) -> bool:
    for character in text.strip():
        if character.isalpha():
            return character.isupper()
    return False


def remove_web_tokens(text: str) -> str:
    text = re.sub(r"\bhttps?://\S+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\S+\.(?:com|org)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\b\S*\.edu\S*\b", "", text, flags=re.IGNORECASE)


def remove_leading_short_lines(
    lines: list[str], max_lines: int = 5, max_words: int = 3
) -> list[str]:
    index = 0
    while index < min(len(lines), max_lines):
        line = lines[index].strip()
        if line and not ends_with_punctuation(line) and len(word_tokens(line)) <= max_words:
            index += 1
        else:
            break
    return lines[index:]


def remove_keyword_headings(lines: Iterable[str]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        tokens = word_tokens(stripped)
        upper_tokens = [token.upper() for token in tokens]
        remove = False
        for keyword in CHAPTER_KEYWORDS:
            if keyword not in upper_tokens:
                continue
            index = upper_tokens.index(keyword)
            if keyword == "PARTIE":
                remove = index <= 3
            else:
                remove = len(tokens) - index - 1 <= 3
            if remove:
                break
        if not remove:
            kept.append(line)
    return kept


def remove_starting_label_lines(lines: Iterable[str]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        tokens = word_tokens(stripped)
        first = tokens[0].lower() if tokens else ""
        starts_label = (
            first in STARTING_LABELS
            or stripped.startswith("第一")
            or (len(tokens) >= 2 and tokens[1].lower() == "parte")
        )
        threshold = 5 if first in {"libro", "traduzione", "trduzione"} else 3
        if starts_label and (
            not ends_with_punctuation(stripped) or len(tokens) <= threshold
        ):
            continue
        kept.append(line)
    return kept


def is_glossary_line(line: str) -> bool:
    stripped = line.strip()
    patterns = (
        r"^\(\d+\)\s+.+[^.!?…]$",
        r"^\d+\.\s+.+[^.!?…]$",
        r"^\d+\.\s+[^:]{0,40}:",
        r"^[IVXLCM]+\.\s+.+[^.!?…]$",
        r"^·\s*.+:\s*.+[^.!?…]$",
        r"^\*\s*.+:\s*.+",
        r"^.\s*:\s*$",
    )
    if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in patterns):
        return True
    if re.match(r"^\d+\.?\s*", stripped) and any(
        character in stripped for character in ";:-"
    ):
        return True
    return stripped.count(":") >= 2 and "." not in stripped and not ends_with_punctuation(stripped)


def is_title_like(line: str, max_words: int = 6) -> bool:
    tokens = word_tokens(line)
    if line.rstrip().endswith(("-", "–", "—")):
        return False
    uppercase_starts = sum(1 for token in tokens if token and token[0].isupper())
    return (
        1 <= len(tokens) <= max_words
        and (
            is_all_caps(line)
            or (
                starts_with_uppercase(line)
                and uppercase_starts >= max(1, (len(tokens) + 1) // 2)
            )
        )
        and not ends_with_punctuation(line)
    )


def remove_title_like_lines(lines: list[str], is_play: bool) -> list[str]:
    if is_play:
        return lines
    kept: list[str] = []
    for index, line in enumerate(lines):
        stripped = strip_trailing_spaces(line)
        previous_empty = index == 0 or not lines[index - 1].strip()
        next_empty = index == len(lines) - 1 or not lines[index + 1].strip()
        surrounded_caps = (
            previous_empty
            and next_empty
            and 1 <= len(word_tokens(stripped)) <= 6
            and is_all_caps(stripped)
        )
        if is_title_like(stripped) or surrounded_caps:
            continue
        kept.append(line)
    return kept


def japanese_short_hiragana_line(line: str, max_tokens: int = 5) -> bool:
    stripped = line.strip()
    if ends_with_punctuation(stripped):
        return False
    tokens = stripped.split()
    return 1 <= len(tokens) <= max_tokens and all(
        HIRAGANA_TOKEN.fullmatch(token) for token in tokens
    )


def filter_lines(lines: list[str], options: TextOptions) -> list[str]:
    lines = remove_leading_short_lines(lines)
    if options.language == "ja":
        lines = [line for line in lines if not japanese_short_hiragana_line(line)]
    if options.remove_headings:
        lines = remove_keyword_headings(lines)
        lines = remove_starting_label_lines(lines)

    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"-\d{1,6}-", stripped):
            continue
        if any(pattern.match(line) for pattern in SCANNER_PATTERNS):
            continue
        if re.fullmatch(r"(?:[.…]\s*){2,}", stripped):
            continue
        if options.remove_glossary_lines and is_glossary_line(line):
            continue
        if (
            options.remove_headings
            and not ends_with_punctuation(stripped)
            and 1 <= len(word_tokens(stripped)) <= 12
            and is_all_caps(stripped)
        ):
            continue
        if options.remove_ocr_fragments and not options.is_play and 0 < len(stripped) <= 5:
            continue
        filtered.append(line)
    return remove_title_like_lines(filtered, options.is_play) if options.remove_headings else filtered


def merge_lowercase_line_starts(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        current = strip_trailing_spaces(lines[index])
        while index + 1 < len(lines):
            following = strip_trailing_spaces(lines[index + 1])
            if (
                following
                and following[0].islower()
                and not ends_with_punctuation(current)
            ):
                if current.endswith("-"):
                    current = current[:-1] + following
                else:
                    current = f"{current} {following}".strip()
                index += 1
            else:
                break
        output.append(current)
        index += 1
    return "\n".join(output)


def remove_digits_touching_letters(text: str) -> str:
    # Python's look-behind cannot express arbitrary Unicode letter classes, so
    # handle mixed alphanumeric tokens with a small replacement function.
    def clean_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(character.isalpha() for character in token):
            return "".join(character for character in token if not character.isdigit())
        return token

    return re.sub(r"[^\W_]+", clean_token, text, flags=re.UNICODE)


def remove_short_hiragana_fillers(line: str) -> str:
    tokens = line.split(" ")
    kept: list[str] = []
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if HIRAGANA_TOKEN.fullmatch(token) and len(token) <= 4:
            if not following or following[0] not in PUNCTUATION_END | {"、"}:
                continue
        kept.append(token)
    return " ".join(kept)


def clean_text(raw: str, options: TextOptions) -> str:
    """Apply the final multilingual text-cleaning sequence."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if options.remove_websites:
        text = remove_web_tokens(text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    lines = filter_lines(text.splitlines(), options)
    text = merge_lowercase_line_starts("\n".join(lines))

    if options.language != "ja":
        text = re.sub(r"([^\W\d_])-\n\s*([^\W\d_])", r"\1\2", text, flags=re.UNICODE)
    text = remove_digits_touching_letters(text)
    text = re.sub(r"\(\d+\)", "", text)
    if options.remove_bracketed_text:
        text = re.sub(r"\[[^]]*\]", "", text, flags=re.DOTALL)
    text = re.sub(r"(?<=[.!?…])\s*\b\d{1,2}\b\s*$", "", text, flags=re.MULTILINE)

    if options.language == "ja":
        text = re.sub(r"[A-Za-z]+", "", text)
        text = "\n".join(remove_short_hiragana_fillers(line) for line in text.splitlines())

    text = text.translate(str.maketrans("", "", "·ㅇ※▪*/"))
    text = re.sub(r"[–—]{2,}", "–", text)
    for _ in range(3):
        text = re.sub(r"^([A-ZÀ-ÖØ-Þ])\s+([A-ZÀ-ÖØ-Þ])\b", r"\1\2", text, flags=re.MULTILINE)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r" {2,}", " ", text)
    return "\n".join(
        strip_trailing_spaces(line) for line in text.splitlines() if line.strip()
    ).strip()


def parse_boolean(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "n"}:
        return False
    if normalized in {"1", "true", "yes", "y"}:
        return True
    raise ValueError(f"Invalid Boolean value: {value!r}")


def ensure_output_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_file(
    input_path: Path,
    output_path: Path,
    options: TextOptions,
    overwrite: bool,
) -> tuple[int, int]:
    ensure_output_available(output_path, overwrite)
    raw = input_path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw, options)
    output_path.write_text(cleaned + ("\n" if cleaned else ""), encoding="utf-8")
    return len(raw), len(cleaned)


def run_text(args: argparse.Namespace) -> None:
    options = TextOptions(
        language=normalize_language(args.language),
        is_play=args.play,
        remove_websites=not args.keep_websites,
        remove_bracketed_text=not args.keep_bracketed_text,
        remove_headings=not args.keep_headings,
        remove_glossary_lines=not args.keep_glossary_lines,
        remove_ocr_fragments=not args.keep_ocr_fragments,
    )
    before, after = clean_file(args.input, args.output, options, args.overwrite)
    print(f"Cleaned {args.input} -> {args.output} ({before:,} -> {after:,} characters)")


def run_batch(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"input_path", "language"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

    processed = 0
    for row_number, row in enumerate(rows, start=2):
        input_path = Path(row["input_path"]).expanduser()
        if not input_path.is_absolute():
            input_path = manifest_path.parent / input_path
        if not input_path.is_file():
            print(f"Skipping row {row_number}: input does not exist: {input_path}", file=sys.stderr)
            continue
        output_name = row.get("output_name", "").strip() or f"{input_path.stem}_cleaned.txt"
        output_path = args.output_directory / output_name
        options = TextOptions(
            language=normalize_language(row["language"]),
            is_play=parse_boolean(row.get("is_play", "")),
        )
        before, after = clean_file(input_path, output_path, options, args.overwrite)
        processed += 1
        print(f"{input_path.name}: {before:,} -> {after:,} characters")
    print(f"Processed {processed} file(s).")


def require_ocr_dependencies():
    try:
        import cv2
        import fitz
        import numpy as np
        import pandas as pd
        import pytesseract
    except ImportError as error:
        raise SystemExit(
            "Japanese PDF extraction requires: pandas numpy pymupdf "
            "opencv-python pytesseract"
        ) from error
    return cv2, fitz, np, pd, pytesseract


def run_japanese_pdf(args: argparse.Namespace) -> None:
    cv2, fitz, np, pd, pytesseract = require_ocr_dependencies()
    tesseract = args.tesseract or shutil.which("tesseract")
    if not tesseract:
        raise SystemExit("Tesseract was not found. Install it or provide --tesseract PATH.")
    pytesseract.pytesseract.tesseract_cmd = str(tesseract)
    if args.tessdata_dir:
        import os
        os.environ["TESSDATA_PREFIX"] = str(args.tessdata_dir.resolve())

    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    combined_path = output_directory / f"{stem}_clean_text.txt"
    audit_path = output_directory / f"{stem}_ocr_audit.csv"
    page_directory = output_directory / f"{stem}_pages"
    ensure_output_available(combined_path, args.overwrite)
    ensure_output_available(audit_path, args.overwrite)
    if page_directory.exists() and any(page_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Page output directory is not empty: {page_directory}")
    page_directory.mkdir(parents=True, exist_ok=True)

    def render_gray(page):
        zoom = args.dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = np.frombuffer(pixmap.samples, np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def mark_ruby(frame):
        if frame.empty:
            frame["is_ruby"] = False
            return frame
        groups = []
        for _, group in frame.groupby(["block_num", "par_num", "line_num"], dropna=False):
            group = group.copy()
            median_height = float(group["height"].median())
            line_top = float(group["top"].min())
            group["is_ruby"] = (
                group["text"].map(lambda text: bool(HIRAGANA_TOKEN.fullmatch(str(text).strip())))
                & (group["height"] <= args.small_ratio * median_height)
                & (
                    group["top"] + group["height"]
                    <= line_top + args.top_fraction * median_height
                )
            )
            groups.append(group)
        return pd.concat(groups, ignore_index=True)

    all_lines: list[str] = []
    audits = []
    document = fitz.open(args.input)
    try:
        for page_number, page in enumerate(document, start=1):
            frame = pytesseract.image_to_data(
                render_gray(page),
                config=f"--psm {args.psm} -l jpn",
                output_type=pytesseract.Output.DATAFRAME,
            )
            frame = frame[(frame.conf != -1) & frame.text.notna()].copy()
            frame["text"] = frame["text"].astype(str).str.strip()
            frame = frame[frame.text != ""]
            for column in (
                "left", "top", "width", "height", "conf", "block_num", "par_num",
                "line_num", "word_num", "page_num", "level",
            ):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["left", "top", "width", "height"])
            frame = mark_ruby(frame)
            frame["page_num"] = page_number
            audits.append(frame.copy())

            page_lines: list[str] = []
            for _, group in frame.groupby(["block_num", "par_num", "line_num"], dropna=False):
                kept = group[~group["is_ruby"]].sort_values(["left", "top", "word_num"])
                if not kept.empty:
                    page_lines.append(re.sub(r"\s+", "", "".join(kept["text"].tolist())))
            (page_directory / f"{stem}_page_{page_number:03d}.txt").write_text(
                "\n".join(page_lines), encoding="utf-8"
            )
            all_lines.extend(page_lines)
            all_lines.append("")
            print(f"Page {page_number}/{document.page_count}: kept {len(page_lines)} lines")
    finally:
        document.close()

    combined_path.write_text("\n".join(all_lines).rstrip() + "\n", encoding="utf-8")
    if audits:
        pd.concat(audits, ignore_index=True).to_csv(audit_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["page_num", "text", "conf", "is_ruby"]).to_csv(
            audit_path, index=False, encoding="utf-8"
        )
    print(f"Combined text: {combined_path}")
    print(f"OCR audit: {audit_path}")


def add_common_text_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--language", required=True, help="language code, e.g. en, fr, es, it, ja")
    parser.add_argument("--play", action="store_true", help="preserve short dialogue/play lines")
    parser.add_argument("--keep-websites", action="store_true")
    parser.add_argument("--keep-bracketed-text", action="store_true")
    parser.add_argument("--keep-headings", action="store_true")
    parser.add_argument("--keep-glossary-lines", action="store_true")
    parser.add_argument("--keep-ocr-fragments", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean multilingual literary texts and OCR Japanese PDFs."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    text_parser = commands.add_parser("text", help="clean one UTF-8 text file")
    text_parser.add_argument("input", type=Path)
    text_parser.add_argument("output", type=Path)
    add_common_text_options(text_parser)
    text_parser.set_defaults(handler=run_text)

    batch_parser = commands.add_parser("batch", help="clean files listed in a CSV manifest")
    batch_parser.add_argument("manifest", type=Path)
    batch_parser.add_argument("output_directory", type=Path)
    batch_parser.add_argument("--overwrite", action="store_true")
    batch_parser.set_defaults(handler=run_batch)

    pdf_parser = commands.add_parser(
        "japanese-pdf", help="OCR horizontal Japanese PDF pages and remove likely furigana"
    )
    pdf_parser.add_argument("input", type=Path)
    pdf_parser.add_argument("output_directory", type=Path)
    pdf_parser.add_argument("--dpi", type=int, default=450)
    pdf_parser.add_argument("--small-ratio", type=float, default=0.75)
    pdf_parser.add_argument("--top-fraction", type=float, default=0.55)
    pdf_parser.add_argument("--psm", type=int, default=6)
    pdf_parser.add_argument("--tesseract", type=Path, help="path to Tesseract executable")
    pdf_parser.add_argument("--tessdata-dir", type=Path, help="directory containing jpn.traineddata")
    pdf_parser.add_argument("--overwrite", action="store_true")
    pdf_parser.set_defaults(handler=run_japanese_pdf)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
