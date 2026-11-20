#!/usr/bin/env python3
"""Reusable literary-text preprocessing. Python 3.10+; standard library only.

INPUT
  UTF-8 .txt files; one text per file.
  Filename: ID_WORK_LANG_VERSION[_AUDIENCE][_OTHER_FIELDS].txt
  ID: digits; WORK: arbitrary label, underscores allowed.
  LANG: lowercase ISO-style code; VERSION: org | adp | abr.
  AUDIENCE: optional; th/theatre indicates dramatic format.
  Trailing checkpoint numbers: accepted, retained unchanged.

RUN (replace uppercase placeholders with your paths)
  python preprocess.py --input INPUT --dry-run --report REVIEW.html
  python preprocess.py --input INPUT --output OUTPUT --config CONFIG.json
  python preprocess.py --print-config > CONFIG.json
  python preprocess.py --help

CONFIGURATION
  JSON sections: defaults, languages, files.
  Precedence: built-ins < defaults < language < file ID.
  Files: keys are digit IDs; leading zeros are optional.
  format: auto | prose | play | verse | preserve.
  Unknown genre: auto assumes prose; override before flattening poetry.
  replacements: exact strings, longest match first, one pass.
  Replacement example: {"言つた": "言った"}; scope to inspected files.
  speaker_names: explicit labels; no name inference.
  full_line_patterns: user regexes; whole-line removal only.

OPTIONAL METADATA (UTF-8 CSV)
  Required: id. Optional: lang, format, genre_org, adp_aud.
  format overrides source genre (e.g. a prose adaptation of a play).
  Missing format: play/theatre or poem/poetry/verse/epic poem inferred
  from genre_org/adp_aud; otherwise prose. Config format overrides CSV.

METHOD
  Structural cleanup precedes whitespace normalisation and exact OCR repairs.
  Plays/verse/preserve retain line structure. General punctuation rules
  still apply unless disabled. Japanese internal-space joining is opt-in.
  Heading/reference deletion and hyphen joining are opt-in.
  Standalone scanner marks, URLs and image filenames are removed by default.
  No blanket lowercasing, spellchecking, bracket-content deletion,
  foreign-script deletion, or unhyphenated word joining.
  Manual correction and selective review remain necessary.

OUTPUT
  Separate directory; original filenames/subdirectories retained.
  Existing files require --overwrite; input files are never overwritten.
  --dry-run writes no cleaned texts; requested reports are still written.
  Optional HTML: per-stage before/after excerpts (not final-text diffs).
  Optional JSON: counts, settings, input/output SHA-256; no excerpts.
  Reports include input filenames; HTML also contains text excerpts.
  This script contains no source corpus, personal identifiers or fixed paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


# 1. Settings and file metadata
DEFAULTS = {
    "format": "auto",
    "remove_source_lines": True,
    "remove_page_lines": False,
    "remove_headings": False,
    "remove_punctuation_lines": False,
    "remove_superscript_digits": False,
    "remove_numeric_references": False,
    "join_hyphenated_breaks": False,
    "normalise_ellipses": True,
    "normalise_punctuation_spaces": True,
    "normalise_whitespace": True,
    "join_japanese_spaces": False,
    "speaker_names": [],
    "replacements": {},
    "full_line_patterns": [],
}
FORMATS = {"auto", "prose", "play", "verse", "preserve"}
FILENAME = re.compile(
    r"^(?P<id>\d+)_(?P<work>.+)_(?P<lang>[a-z]{2,3})_"
    r"(?P<version>org|adp|abr)(?:_(?P<suffix>.+))?\.txt$"
)
H = r"[^\S\n]"
LETTER = r"[^\W\d_]"
JP = (r"\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FC-\u30FF"
      r"\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
      r"\U00020000-\U0002FA1F々〆〇")
HEADING_WORDS = {
    "en": r"chapter|part|book|volume|prologue|epilogue",
    "fr": r"chapitre|partie|livre|tome|prologue|épilogue",
    "es": r"capítulo|parte|libro|volumen|prólogo|epílogo",
    "it": r"capitolo|parte|libro|volume|prologo|epilogo",
    "bg": r"глава|част|том|пролог|епилог",
    "ru": r"глава|часть|том|пролог|эпилог",
    "el": r"κεφάλαιο|μέρος|τόμος|πρόλογος|επίλογος",
}


def normal_id(value):
    value = str(value).strip()
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"Invalid numeric ID: {value!r}")
    return str(int(value))


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object")
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    for key, value in settings.items():
        if isinstance(DEFAULTS[key], bool) and type(value) is not bool:
            raise ValueError(f"{key}: expected true/false")
    if settings.get("format", "auto") not in FORMATS:
        raise ValueError("Invalid format setting")
    for key in ("speaker_names", "full_line_patterns"):
        value = settings.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(s, str) or not s.strip() or "\n" in s for s in value
        ):
            raise ValueError(f"{key}: expected nonempty single-line strings")
    replacements = settings.get("replacements", {})
    if not isinstance(replacements, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str)
        for k, v in replacements.items()
    ):
        raise ValueError("replacements: expected nonempty string keys/string values")
    for pattern in settings.get("full_line_patterns", []):
        re.compile(pattern)


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path else {}
    if not isinstance(config, dict) or set(config) - {"defaults", "languages", "files"}:
        raise ValueError("Config sections: defaults, languages, files only")
    validate_settings(config.get("defaults", {}))
    for section in ("languages", "files"):
        if not isinstance(config.get(section, {}), dict):
            raise ValueError(f"{section}: expected an object")
        for settings in config.get(section, {}).values():
            if not isinstance(settings, dict):
                raise ValueError(f"{section}: entries must be objects")
            validate_settings(settings)
    files = {}
    for key, value in config.get("files", {}).items():
        ident = normal_id(key)
        if ident in files:
            raise ValueError(f"Duplicate config ID: {key}")
        files[ident] = value
    config["files"] = files
    return config


def load_metadata(path):
    if path is None:
        return {}
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError("Metadata requires an id column")
        for row in reader:
            ident = normal_id(row["id"])
            if ident in rows:
                raise ValueError(f"Duplicate metadata ID: {ident}")
            rows[ident] = {k: (v or "").strip() for k, v in row.items()}
    return rows


def settings_for(info, row, config):
    settings = dict(DEFAULTS)
    for overrides in (config.get("defaults", {}),
                      config.get("languages", {}).get(info["lang"], {}),
                      config.get("files", {}).get(info["id"], {})):
        settings.update(overrides)  # Lists/dictionaries replace rather than merge.
    if row.get("lang") and row["lang"] != info["lang"]:
        raise ValueError(f"Language mismatch for ID {info['id']}")
    if settings["format"] == "auto":
        explicit = row.get("format", "").lower()
        if explicit and explicit not in FORMATS:
            raise ValueError(f"Invalid metadata format for ID {info['id']}")
        if explicit and explicit != "auto":
            settings["format"] = explicit
        else:
            tags = set((info.get("suffix") or "").lower().split("_"))
            genre = row.get("genre_org", "").lower()
            audience = row.get("adp_aud", "").lower()
            if genre in {"play", "drama", "theatre"} or audience in {"th", "theatre"} or tags & {"th", "theatre"}:
                settings["format"] = "play"
            elif genre in {"poem", "poetry", "verse", "epic poem"}:
                settings["format"] = "verse"
            else:
                settings["format"] = "prose"
    return settings


# 2. Ordered transformations and bounded stage-level previews
class Cleaner:
    def __init__(self, text, preview_limit=5):
        self.text = text
        self.counts = Counter()
        self.examples = {}
        self.preview_limit = preview_limit

    def sub(self, stage, pattern, replacement, flags=0):
        before = self.text
        samples = self.examples.setdefault(stage, [])

        def replace(match):
            after = replacement(match) if callable(replacement) else replacement
            if after != match.group():
                self.counts[stage] += 1
                if len(samples) < self.preview_limit:
                    a, b = match.span()
                    # Bound large deletions as well as surrounding context.
                    def excerpt(s):
                        return s if len(s) <= 400 else s[:400] + " […]"
                    left, right = before[max(0, a-70):a], before[b:b+70]
                    samples.append({
                        "before": left + "⟦" + excerpt(match.group()) + "⟧" + right,
                        "after": left + "⟦" + excerpt(after) + "⟧" + right,
                    })
            return after

        self.text = re.sub(pattern, replace, before, flags=flags)


def preprocess(text, language, settings, preview_limit=5):
    """Return (cleaned_text, change_counts, per_stage_examples). No file writes."""
    validate_settings(settings)
    if settings["format"] == "auto":
        raise ValueError("Resolve format before calling preprocess()")
    c = Cleaner(text, preview_limit)
    prose = settings["format"] == "prose"
    c.sub("encoding and line breaks", r"\A\ufeff|\r\n?|\f",
          lambda m: "" if m.group() == "\ufeff" else "\n")

    if settings["remove_source_lines"]:
        def source_line(m):
            s = m.group().strip()
            tokens = s.casefold().split()
            scanner = "camscanner" in tokens and all(
                t in {"camscanner", "cs", "scanned", "with"} for t in tokens)
            url = re.fullmatch(r"(?:https?://|www\.)\S+", s, re.I)
            image = re.fullmatch(r"\S+\.(?:jpe?g|png|gif|tiff?|bmp|webp)(?:\s+\([^\n)]{1,30}\))?", s, re.I)
            return "" if scanner or url or image else m.group()
        c.sub("standalone source artifacts", r"^[^\n]+", source_line, re.M)

    for pattern in settings["full_line_patterns"]:
        c.sub("custom full-line removal", r"^[^\n]+",
              lambda m, p=pattern: "" if re.fullmatch(p, m.group().strip()) else m.group(), re.M)

    if settings["remove_page_lines"]:
        c.sub("standalone numeric lines", rf"^{H}*(?:[0-9]+|\[[0-9]+\]|\([0-9]+\)){H}*$", "", re.M)
    if settings["remove_headings"] and prose and language in HEADING_WORDS:
        # Explicit numbered headings only: no guessed title-like narrative.
        label = HEADING_WORDS[language]
        c.sub("explicit numbered headings", rf"^{H}*(?:{label}){H}+(?:[0-9]+|[IVXLCDM]+)\.?{H}*$", "", re.M | re.I)
    if settings["remove_punctuation_lines"] and prose:
        def punctuation_only(m):
            chars = [ch for ch in m.group() if not ch.isspace()]
            return "" if chars and all(unicodedata.category(ch).startswith("P") for ch in chars) else m.group()
        c.sub("punctuation-only lines", r"^[^\n]+", punctuation_only, re.M)
    if settings["remove_numeric_references"]:
        c.sub("numeric reference markers", r"[0-9]+\^\{[0-9]+\}|\^\{[0-9]+\}|\([0-9]+\)|\[[0-9]+\]", "")
    if settings["remove_superscript_digits"]:
        c.sub("superscript digits", r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", "")

    if settings["join_hyphenated_breaks"] and prose and language != "ja":
        # Keep spaced dashes and ordinary compounds. Ambiguous compounds
        # broken across lines can still be misjoined: enable after review.
        pattern = rf"(?P<a>{LETTER})[-\u00ad]{H}*\n{H}*(?P<b>{LETTER})"
        while True:
            before = c.text
            c.sub("hyphenated line breaks", pattern,
                  lambda m: m["a"] + m["b"] if m["b"].islower() else m.group())
            if c.text == before:
                break
        c.sub("soft hyphens", "\u00ad", "")

    if settings["normalise_ellipses"]:
        c.sub("ellipses", rf"[.…](?:{H}*[.…])*",
              lambda m: "..." if sum(3 if x == "…" else x == "." for x in m.group()) >= 3 else m.group())
    if settings["normalise_punctuation_spaces"]:
        punct = ",.，。、"
        if language not in {"fr", "ja"}:
            punct += ";:!?"
        c.sub("spaces before punctuation", rf"(?<=[^\W_]){H}{{1,3}}(?=[{re.escape(punct)}])", "")

    if settings["format"] == "play" and settings["speaker_names"]:
        names = "|".join(re.escape(s) for s in sorted(settings["speaker_names"], key=len, reverse=True))
        c.sub("speaker labels", rf"^{H}*(?P<name>{names})(?:[.:]{H}*|{H}+|(?=$))",
              lambda m: m["name"] + ":" + ("\n" if m.end() < len(c.text) and c.text[m.end()] != "\n" else ""), re.M)

    if settings["normalise_whitespace"]:
        if prose:
            c.sub("prose whitespace", r"\s+", " ")
            c.sub("outer prose whitespace", r"\A +| +\Z", "")
        else:
            c.sub("trailing line whitespace", r"[^\S\n]+$", "", re.M)
    if language == "ja" and settings["join_japanese_spaces"]:
        # Removes intentional pedagogical word spacing too. No word inference.
        gap = r"\s+" if prose else rf"{H}+"
        c.sub("Japanese internal whitespace", rf"(?<=[{JP}]){gap}(?=[{JP}])", "")
    if settings["replacements"]:
        mapping = settings["replacements"]
        c.sub("explicit OCR replacements", "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)),
              lambda m: mapping[m.group()])
    if not c.text.strip():
        raise ValueError("Preprocessing would produce an empty text")
    return c.text, dict(c.counts), {k: v for k, v in c.examples.items() if v}


# 3. Optional local HTML review (no JavaScript or external services)
def html_report(records):
    sections = []
    for r in records:
        stages = []
        for stage in r["counts"]:
            examples = r["examples"].get(stage, [])
            panels = "".join(
                '<div class="pair"><pre>' + html.escape(e["before"]) + '</pre><pre>'
                + html.escape(e["after"]) + '</pre></div>' for e in examples)
            stages.append("<details><summary>" + html.escape(stage) + " — "
                          + str(r["counts"][stage]) + " changes</summary>" + panels + "</details>")
        sections.append("<section><h2>" + html.escape(r["file"]) + "</h2><p>Format: "
                        + html.escape(r["settings"]["format"]) + "</p>"
                        + ("".join(stages) or "<p>No recorded changes.</p>") + "</section>")
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Preprocessing review</title><style>
body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}
section{border-top:1px solid #bbb;margin-top:25px}summary{cursor:pointer;padding:10px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f4f6;padding:12px;font:inherit}
@media(max-width:650px){.pair{grid-template-columns:1fr}}
</style><h1>Preprocessing review</h1><p>Left: before. Right: after.
Brackets mark changes. Excerpts are stage-local, limited samples; later stages
may change them further. Counts are replacements, not characters. Large changes
are truncated. Review is not exhaustive.</p>''' + "".join(sections) + "</html>"


# 4. Command-line workflow: validate, process, then write separate outputs
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Text file or directory (recursive)")
    parser.add_argument("--output", type=Path, help="Separate output directory")
    parser.add_argument("--metadata", type=Path, help="Optional CSV; see header")
    parser.add_argument("--config", type=Path, help="Optional JSON; see header")
    parser.add_argument("--report", type=Path, help="Optional HTML review path")
    parser.add_argument("--log", type=Path, help="Optional JSON audit path")
    parser.add_argument("--samples", type=int, default=5, help="Excerpts per stage/file; default 5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-config", action="store_true", help="Print default JSON and exit")
    args = parser.parse_args(argv)
    if args.print_config:
        print(json.dumps({"defaults": DEFAULTS, "languages": {}, "files": {}}, ensure_ascii=False, indent=2))
        return 0
    if not args.input or (not args.dry_run and not args.output):
        parser.error("--input required; --output required unless --dry-run")
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    try:
        root = args.input.resolve(strict=True)
        config, metadata = load_config(args.config), load_metadata(args.metadata)
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt") if root.is_dir() else [root]
        if not paths:
            raise ValueError("No text files found")
        if any(p.is_symlink() for p in paths):
            raise ValueError("Use regular input files, not symbolic links")
        output = args.output.resolve() if args.output else None
        if output and root.is_dir() and (output == root or root in output.parents or output in root.parents):
            raise ValueError("Input and output directories must not contain one another")
        plan, identifiers = [], set()
        for path in paths:
            match = FILENAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Filename does not match documented format: {path.name}")
            info = match.groupdict()
            info["id"] = normal_id(info["id"])
            if info["id"] in identifiers:
                raise ValueError(f"Duplicate ID: {info['id']}")
            identifiers.add(info["id"])
            relative = path.relative_to(root) if root.is_dir() else Path(path.name)
            settings = settings_for(info, metadata.get(info["id"], {}), config)
            plan.append((path, relative, info, settings))
        unknown = set(config.get("files", {})) - identifiers
        if unknown:
            raise ValueError(f"Config IDs not present in input: {sorted(unknown)}")
        targets = ([output / rel for _, rel, _, _ in plan] if output and not args.dry_run else [])
        targets += [p.resolve() for p in (args.report, args.log) if p]
        protected = {p.resolve() for p in paths}
        protected |= {p.resolve() for p in (args.config, args.metadata) if p}
        resolved_targets = [p.resolve() for p in targets]
        if len(set(resolved_targets)) != len(resolved_targets):
            raise ValueError("Output/report/log paths collide")
        for target in resolved_targets:
            if target in protected or (root.is_dir() and root in target.parents):
                raise ValueError("Outputs must not overwrite or be stored inside inputs")
            if target.exists() and (not args.overwrite or not target.is_file()):
                raise ValueError(f"Output exists; use --overwrite for files: {target.name}")
        records, cleaned_files = [], []
        for path, relative, info, settings in plan:
            raw = path.read_bytes()
            original = raw.decode("utf-8", errors="strict")
            cleaned, counts, examples = preprocess(original, info["lang"], settings, args.samples if args.report else 0)
            data = cleaned.encode("utf-8")
            records.append({"file": relative.as_posix(), "id": info["id"], "lang": info["lang"],
                            "settings": settings, "counts": counts, "examples": examples,
                            "input_sha256": hashlib.sha256(raw).hexdigest(),
                            "output_sha256": hashlib.sha256(data).hexdigest()})
            cleaned_files.append((relative, data))
        # No writes until every input has decoded and processed successfully.
        if not args.dry_run:
            for relative, data in cleaned_files:
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(html_report(records), encoding="utf-8")
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = [{k: v for k, v in r.items() if k != "examples"} for r in records]
            args.log.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        totals = Counter()
        for record in records:
            totals.update(record["counts"])
        print(f"Processed {len(records)} texts; {'preview only' if args.dry_run else 'outputs written'}.")
        print("Resolved formats:", dict(Counter(r["settings"]["format"] for r in records)))
        for stage, count in totals.items():
            print(f"  {stage}: {count:,}")
        return 0
    except (OSError, ValueError, TypeError, re.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
