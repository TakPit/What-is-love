#!/usr/bin/env python3
from __future__ import annotations

"""
Prepare, bin, and sanity-check lyric corpora in one pass.

This script merges the current phase-1 corpus preparation and light exploration
steps into a single entry point:
1) read raw Genius parquet shards,
2) apply the current conservative cleaning/tokenization,
3) assign each song to a user-defined time bin,
4) write one gzipped tokenized sentence file per bin,
5) save compact, training-oriented summaries and sanity reports.

The binning is defined by a list of increasing year edges, e.g.
  1960,1970,1980,1990,2000,2010,2020
which creates half-open bins:
  [1960,1970), [1970,1980), ... [2010,2020)
with output files named:
  1960_1969.txt.gz, 1970_1979.txt.gz, ... , 2010_2019.txt.gz
"""

import argparse
import csv
import gzip
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import polars as pl

REQUIRED_COLUMNS = ["title", "artist", "tag", "year", "lyrics"]

HEADER_LINE_RE = re.compile(r"^\[[^\]]+\]\s*$")
INLINE_BRACKET_RE = re.compile(r"\[[^\]]+\]")
MULTISPACE_RE = re.compile(r"\s+")
QUOTE_BOUNDARY_RE = re.compile(r'"\s+"')
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
REPEATED_CHAR_RE = re.compile(r"(.)\1\1+")
APOSTROPHE_RE = re.compile(r"'")
ASCII_ALPHA_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")

TRANSLATION_TABLE = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
})

DEFAULT_TARGETS = [
    "love", "heart", "baby", "kiss", "touch",
    "forever", "mine", "desire", "hurt", "broken",
]
DEFAULT_THRESHOLDS = [5, 10, 20, 50]
DEFAULT_EXACT_FILLERS = [
    "oh", "ooh", "oooh", "ooooh", "ah", "aah", "aaah", "uh", "uhh",
    "huh", "hm", "hmm", "mmm", "mm", "nah", "na", "la", "lalala",
    "woah", "whoa", "yeah", "yea", "yeh", "yo", "hey", "ayy", "aye",
    "ha", "hah", "hee", "ho",
]

FILLER_REMOVAL_TOKENS = {
    "ah", "aah", "aaah", "eh", "ha", "hey", "hm", "hmm", "hmmm", "huh",
    "la", "mm", "mmm", "na", "nah", "oh", "ooh", "uh", "uhh", "whoa",
    "woah", "ye", "yea", "yeah", "yo", "ahh", "ahhh", "haa", "haaah",
    "hah", "haha", "heh", "hmhm", "hmmh", "laa", "laaa", "naa", "naaa",
    "nahh", "ohh", "ohhh", "oohh", "oooh", "ooooh", "uhhh", "whoooa",
    "woooah", "yeahh", "yeahhh", "aye", "ayy",
}
ONE_CHAR_TOKEN_WHITELIST = {"i", "a"}
REPEATED_FILLER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^y+e+a+h+$"), "yeah"),
    (re.compile(r"^y+e+a+$"), "yea"),
    (re.compile(r"^a+h+$"), "ah"),
    (re.compile(r"^o+h+$"), "oh"),
    (re.compile(r"^o+u+h+$"), "ooh"),
    (re.compile(r"^u+h+$"), "uh"),
    (re.compile(r"^n+a+h*$"), "na"),
    (re.compile(r"^l+a+$"), "la"),
    (re.compile(r"^h+a+$"), "ha"),
    (re.compile(r"^h+m+$"), "hmm"),
    (re.compile(r"^m+$"), "mmm"),
    (re.compile(r"^w+h+o+a+$"), "whoa"),
    (re.compile(r"^w+o+a+h+$"), "woah"),
    (re.compile(r"^y+o+$"), "yo"),
    (re.compile(r"^h+e+y+$"), "hey"),
    (re.compile(r"^a+y+e*$"), "aye"),
]


class BinSpec(tuple):
    @property
    def start(self) -> int:
        return self[0]

    @property
    def end(self) -> int:
        return self[1]

    @property
    def label(self) -> str:
        return self[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and sanity-check lyric corpora for arbitrary year bins in one pass."
        )
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing parquet shards, or a single parquet file.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Run directory under results/phase1/corpus_preparation_and_exploration/<bin-id>")
    parser.add_argument("--bin-edges", type=str, required=True,
                        help="Comma-separated increasing year edges, e.g. 1960,1970,1980,1990,2000,2010,2020")
    parser.add_argument("--targets", type=str, default=",".join(DEFAULT_TARGETS),
                        help="Comma-separated target words to track by bin.")
    parser.add_argument("--thresholds", type=str, default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
                        help="Comma-separated vocabulary thresholds, e.g. 5,10,20,50")
    parser.add_argument("--top-n", type=int, default=100,
                        help="How many top tokens to save per bin in top_tokens_by_bin.json")
    parser.add_argument("--sample-n", type=int, default=0,
                        help="If > 0, stop after this many valid songs inside the requested year range.")
    parser.add_argument("--dedup-adjacent-lines", action="store_true",
                        help="Remove adjacent identical cleaned lines within each song.")
    return parser.parse_args()


def discover_parquet_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {path}")
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return files


def parse_csv_list(text: str) -> list[str]:
    return [part.strip().lower() for part in text.split(",") if part.strip()]


def parse_thresholds(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("At least one threshold is required.")
    return sorted(set(vals))


def parse_bin_edges(text: str) -> list[BinSpec]:
    edges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        edges.append(int(part))
    if len(edges) < 2:
        raise ValueError("At least two bin edges are required.")
    if edges != sorted(edges) or len(set(edges)) != len(edges):
        raise ValueError("Bin edges must be strictly increasing.")

    bins: list[BinSpec] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            raise ValueError("Each bin edge must be greater than the previous one.")
        label = f"{start}_{end - 1}"
        bins.append(BinSpec((start, end, label)))
    return bins


def assign_bin(year: int, bins: list[BinSpec]) -> BinSpec | None:
    for spec in bins:
        if spec.start <= year < spec.end:
            return spec
    return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.translate(TRANSLATION_TABLE)


def normalize_repeated_filler_token(token: str) -> tuple[str, bool]:
    for pattern, canonical in REPEATED_FILLER_PATTERNS:
        if pattern.fullmatch(token):
            return canonical, canonical != token
    return token, False


def filter_and_normalize_tokens(tokens: list[str]) -> tuple[list[str], int, int, int]:
    filtered: list[str] = []
    repeated_fillers_normalized = 0
    fillers_removed = 0
    one_char_tokens_dropped = 0

    for token in tokens:
        token, changed = normalize_repeated_filler_token(token)
        if changed:
            repeated_fillers_normalized += 1

        if token in FILLER_REMOVAL_TOKENS:
            fillers_removed += 1
            continue

        if len(token) == 1 and token not in ONE_CHAR_TOKEN_WHITELIST:
            one_char_tokens_dropped += 1
            continue

        filtered.append(token)

    return filtered, repeated_fillers_normalized, fillers_removed, one_char_tokens_dropped


def prepare_song(lyrics: str, dedup_adjacent_lines: bool) -> dict:
    text = normalize_text(lyrics)
    text = QUOTE_BOUNDARY_RE.sub("\n", text)
    text = text.replace('"', "")

    raw_nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    headers_removed = 0
    cleaned_lines = []
    for line in raw_nonempty_lines:
        if HEADER_LINE_RE.fullmatch(line):
            headers_removed += 1
            continue
        line = INLINE_BRACKET_RE.sub(" ", line)
        line = line.replace('"', "")
        line = MULTISPACE_RE.sub(" ", line).strip()
        if line:
            cleaned_lines.append(line)

    adjacent_dups_removed = 0
    if dedup_adjacent_lines:
        deduped_lines = []
        last_key = None
        for line in cleaned_lines:
            key = line.casefold()
            if key == last_key:
                adjacent_dups_removed += 1
                continue
            deduped_lines.append(line)
            last_key = key
    else:
        deduped_lines = cleaned_lines

    tokenized_lines = []
    token_count = 0
    repeated_fillers_normalized = 0
    fillers_removed = 0
    one_char_tokens_dropped = 0

    for line in deduped_lines:
        raw_tokens = TOKEN_RE.findall(line.lower())
        if not raw_tokens:
            continue
        tokens, n_norm, n_fillers, n_onechar = filter_and_normalize_tokens(raw_tokens)
        repeated_fillers_normalized += n_norm
        fillers_removed += n_fillers
        one_char_tokens_dropped += n_onechar
        if tokens:
            tokenized_lines.append(tokens)
            token_count += len(tokens)

    return {
        "raw_nonempty_lines": len(raw_nonempty_lines),
        "clean_lines": len(cleaned_lines),
        "final_lines": len(deduped_lines),
        "headers_removed": headers_removed,
        "adjacent_dups_removed": adjacent_dups_removed,
        "tokenized_lines": tokenized_lines,
        "token_line_count": len(tokenized_lines),
        "token_count": token_count,
        "repeated_fillers_normalized": repeated_fillers_normalized,
        "fillers_removed": fillers_removed,
        "one_char_tokens_dropped": one_char_tokens_dropped,
    }


def safe_year(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def open_bin_writer(base_dir: Path, label: str, cache: dict[str, gzip.GzipFile]) -> gzip.GzipFile:
    if label not in cache:
        cache[label] = gzip.open(base_dir / f"{label}.txt.gz", "wt", encoding="utf-8")
    return cache[label]


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def rate_per_million(count: int, total: int) -> float:
    return round((count / total) * 1_000_000, 3) if total else 0.0


def compute_bin_sanity(token_counter: Counter[str], line_lengths: list[int], exact_fillers: set[str]) -> dict:
    token_total = int(sum(token_counter.values()))
    vocab_size = len(token_counter)
    hapax_count = sum(1 for _, c in token_counter.items() if c == 1)
    repeated_char_total = sum(c for tok, c in token_counter.items() if REPEATED_CHAR_RE.search(tok))
    apostrophe_total = sum(c for tok, c in token_counter.items() if APOSTROPHE_RE.search(tok))
    malformed_total = sum(c for tok, c in token_counter.items() if ASCII_ALPHA_RE.fullmatch(tok) is None)
    exact_filler_left_total = sum(c for tok, c in token_counter.items() if tok in exact_fillers)
    return {
        "token_total": token_total,
        "vocab_size": vocab_size,
        "hapax_count": hapax_count,
        "hapax_share": round(hapax_count / vocab_size, 4) if vocab_size else 0.0,
        "ttr_per_1k": round(vocab_size / token_total * 1000.0, 4) if token_total else 0.0,
        "line_count": len(line_lengths),
        "median_line_len": statistics.median(line_lengths) if line_lengths else 0,
        "p90_line_len": round(percentile(line_lengths, 0.90), 2),
        "p99_line_len": round(percentile(line_lengths, 0.99), 2),
        "max_line_len": max(line_lengths) if line_lengths else 0,
        "repeated_char_total": repeated_char_total,
        "apostrophe_total": apostrophe_total,
        "malformed_total": malformed_total,
        "exact_filler_left_total": exact_filler_left_total,
    }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_files = discover_parquet_files(args.input_dir)
    bins = parse_bin_edges(args.bin_edges)
    targets = parse_csv_list(args.targets)
    thresholds = parse_thresholds(args.thresholds)
    exact_fillers = set(DEFAULT_EXACT_FILLERS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences_dir = output_dir / "sentences_by_bin"
    sentences_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, gzip.GzipFile] = {}
    stop_early = False
    valid_in_range_rows = 0

    bin_order = [spec.label for spec in bins]
    bin_preparation = {
        spec.label: {
            "bin_start": spec.start,
            "bin_end_exclusive": spec.end,
            "songs": 0,
            "raw_nonempty_lines": 0,
            "tokenized_lines": 0,
            "tokens": 0,
            "headers_removed": 0,
            "adjacent_dups_removed": 0,
            "repeated_fillers_normalized": 0,
            "fillers_removed": 0,
            "one_char_tokens_dropped": 0,
        }
        for spec in bins
    }
    token_counts_by_bin: dict[str, Counter[str]] = {spec.label: Counter() for spec in bins}
    line_lengths_by_bin: dict[str, list[int]] = {spec.label: [] for spec in bins}

    summary = {
        "input_files_seen": 0,
        "rows_seen": 0,
        "rows_missing_required": 0,
        "rows_invalid_year": 0,
        "rows_empty_lyrics": 0,
        "rows_outside_requested_bins": 0,
        "rows_in_requested_bins": 0,
        "rows_kept_for_training": 0,
        "global_repeated_fillers_normalized": 0,
        "global_filler_tokens_removed": 0,
        "global_one_char_tokens_dropped": 0,
    }

    for parquet_file in input_files:
        summary["input_files_seen"] += 1
        df = pl.read_parquet(parquet_file, columns=REQUIRED_COLUMNS, use_pyarrow=True)
        for row in df.iter_rows(named=True):
            summary["rows_seen"] += 1
            title = row.get("title")
            artist = row.get("artist")
            tag = row.get("tag")
            year = safe_year(row.get("year"))
            lyrics = row.get("lyrics")

            if title is None or artist is None or tag is None or lyrics is None:
                summary["rows_missing_required"] += 1
                continue
            lyrics = str(lyrics)
            if not lyrics.strip():
                summary["rows_empty_lyrics"] += 1
                continue
            if year is None:
                summary["rows_invalid_year"] += 1
                continue

            spec = assign_bin(year, bins)
            if spec is None:
                summary["rows_outside_requested_bins"] += 1
                continue
            summary["rows_in_requested_bins"] += 1
            valid_in_range_rows += 1

            prepared = prepare_song(lyrics=lyrics, dedup_adjacent_lines=args.dedup_adjacent_lines)
            summary["global_repeated_fillers_normalized"] += prepared["repeated_fillers_normalized"]
            summary["global_filler_tokens_removed"] += prepared["fillers_removed"]
            summary["global_one_char_tokens_dropped"] += prepared["one_char_tokens_dropped"]

            if prepared["token_line_count"] == 0:
                if args.sample_n > 0 and valid_in_range_rows >= args.sample_n:
                    stop_early = True
                    break
                continue

            summary["rows_kept_for_training"] += 1
            prep = bin_preparation[spec.label]
            prep["songs"] += 1
            prep["raw_nonempty_lines"] += prepared["raw_nonempty_lines"]
            prep["tokenized_lines"] += prepared["token_line_count"]
            prep["tokens"] += prepared["token_count"]
            prep["headers_removed"] += prepared["headers_removed"]
            prep["adjacent_dups_removed"] += prepared["adjacent_dups_removed"]
            prep["repeated_fillers_normalized"] += prepared["repeated_fillers_normalized"]
            prep["fillers_removed"] += prepared["fillers_removed"]
            prep["one_char_tokens_dropped"] += prepared["one_char_tokens_dropped"]

            writer = open_bin_writer(sentences_dir, spec.label, writers)
            for tokens in prepared["tokenized_lines"]:
                writer.write(" ".join(tokens) + "\n")
                token_counts_by_bin[spec.label].update(tokens)
                line_lengths_by_bin[spec.label].append(len(tokens))

            if args.sample_n > 0 and valid_in_range_rows >= args.sample_n:
                stop_early = True
                break
        if stop_early:
            break

    for fh in writers.values():
        fh.close()

    run_config = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "bin_edges": [spec.start for spec in bins] + [bins[-1].end],
        "bins": [
            {"label": spec.label, "start": spec.start, "end_exclusive": spec.end, "end_inclusive": spec.end - 1}
            for spec in bins
        ],
        "targets": targets,
        "thresholds": thresholds,
        "dedup_adjacent_lines": bool(args.dedup_adjacent_lines),
        "sample_n": args.sample_n,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2, ensure_ascii=False)

    bin_summary_rows = []
    target_rows = []
    threshold_rows = []
    top_tokens_payload: dict[str, list[list[object]]] = {}
    threshold_sets: dict[int, dict[str, set[str]]] = {thr: {} for thr in thresholds}

    for spec in bins:
        label = spec.label
        token_counter = token_counts_by_bin[label]
        line_lengths = line_lengths_by_bin[label]
        sanity = compute_bin_sanity(token_counter, line_lengths, exact_fillers)
        prep = bin_preparation[label]

        bin_summary_rows.append({
            "bin_label": label,
            "start_year": spec.start,
            "end_year": spec.end - 1,
            **prep,
            **sanity,
            "repeated_char_per_million": rate_per_million(sanity["repeated_char_total"], sanity["token_total"]),
            "apostrophe_per_million": rate_per_million(sanity["apostrophe_total"], sanity["token_total"]),
            "malformed_per_million": rate_per_million(sanity["malformed_total"], sanity["token_total"]),
        })

        for target in targets:
            count = int(token_counter.get(target, 0))
            target_rows.append({
                "bin_label": label,
                "start_year": spec.start,
                "end_year": spec.end - 1,
                "target": target,
                "count": count,
                "per_million_tokens": rate_per_million(count, sanity["token_total"]),
            })

        for thr in thresholds:
            vocab_at_or_above = {tok for tok, count in token_counter.items() if count >= thr}
            threshold_sets[thr][label] = vocab_at_or_above
            threshold_rows.append({
                "bin_label": label,
                "start_year": spec.start,
                "end_year": spec.end - 1,
                "threshold": thr,
                "vocab_at_or_above_threshold": len(vocab_at_or_above),
            })

        top_tokens_payload[label] = [[tok, int(cnt)] for tok, cnt in token_counter.most_common(args.top_n)]

    shared_rows = []
    for thr in thresholds:
        sets = [threshold_sets[thr][label] for label in bin_order]
        intersection = set.intersection(*sets) if sets else set()
        shared_rows.append({
            "threshold": thr,
            "bins_considered": len(sets),
            "shared_vocab_all_bins": len(intersection),
        })

    write_csv(
        output_dir / "bin_summary.csv",
        bin_summary_rows,
        [
            "bin_label", "start_year", "end_year", "bin_start", "bin_end_exclusive",
            "songs", "raw_nonempty_lines", "tokenized_lines", "tokens", "headers_removed",
            "adjacent_dups_removed", "repeated_fillers_normalized", "fillers_removed",
            "one_char_tokens_dropped", "token_total", "vocab_size", "hapax_count",
            "hapax_share", "ttr_per_1k", "line_count", "median_line_len", "p90_line_len",
            "p99_line_len", "max_line_len", "repeated_char_total", "apostrophe_total",
            "malformed_total", "exact_filler_left_total", "repeated_char_per_million",
            "apostrophe_per_million", "malformed_per_million",
        ],
    )
    write_csv(
        output_dir / "target_word_counts_by_bin.csv",
        target_rows,
        ["bin_label", "start_year", "end_year", "target", "count", "per_million_tokens"],
    )
    write_csv(
        output_dir / "threshold_coverage_by_bin.csv",
        threshold_rows,
        ["bin_label", "start_year", "end_year", "threshold", "vocab_at_or_above_threshold"],
    )
    write_csv(
        output_dir / "shared_vocab_intersection.csv",
        shared_rows,
        ["threshold", "bins_considered", "shared_vocab_all_bins"],
    )
    with open(output_dir / "top_tokens_by_bin.json", "w", encoding="utf-8") as fh:
        json.dump(top_tokens_payload, fh, indent=2, ensure_ascii=False)
    with open(output_dir / "preparation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    tokens_by_label = {row["bin_label"]: row["token_total"] for row in bin_summary_rows}
    warnings: list[str] = []
    empty_bins = [label for label in bin_order if tokens_by_label.get(label, 0) == 0]
    if empty_bins:
        warnings.append(f"Empty bins: {', '.join(empty_bins)}")

    nonempty_items = [(label, tokens) for label, tokens in tokens_by_label.items() if tokens > 0]
    if nonempty_items:
        min_label, min_tokens = min(nonempty_items, key=lambda x: x[1])
        max_label, max_tokens = max(nonempty_items, key=lambda x: x[1])
        ratio = max_tokens / min_tokens if min_tokens else None
        if ratio is not None and ratio >= 5:
            warnings.append(
                f"Token imbalance across bins is high ({max_label} / {min_label} = {ratio:.2f})."
            )
    else:
        min_label = max_label = None
        min_tokens = max_tokens = 0
        ratio = None

    missing_targets = {}
    for target in targets:
        missing = [label for label in bin_order if token_counts_by_bin[label].get(target, 0) == 0]
        if missing:
            missing_targets[target] = missing
    if missing_targets:
        warnings.append("Some target words are absent from one or more bins.")

    sanity_summary = {
        "bin_labels": bin_order,
        "warnings": warnings,
        "missing_targets": missing_targets,
        "token_imbalance": {
            "min_bin": min_label,
            "min_tokens": min_tokens,
            "max_bin": max_label,
            "max_tokens": max_tokens,
            "max_over_min_ratio": round(ratio, 4) if ratio else None,
        },
        "threshold_summary": {
            str(thr): {
                "shared_vocab_all_bins": next(row["shared_vocab_all_bins"] for row in shared_rows if row["threshold"] == thr),
                "per_bin_vocab": {
                    row["bin_label"]: row["vocab_at_or_above_threshold"]
                    for row in threshold_rows if row["threshold"] == thr
                },
            }
            for thr in thresholds
        },
        "interpretation_notes": [
            "Use bin_summary.csv to compare corpus size and token-quality diagnostics across the requested bins.",
            "Use target_word_counts_by_bin.csv to check whether key analysis words remain available after rebinnig.",
            "Use threshold_coverage_by_bin.csv and shared_vocab_intersection.csv to anticipate stability for later embedding alignment and focused analysis.",
            "The gzipped files in sentences_by_bin/ are the training-ready text corpora for the same bin specification.",
        ],
    }
    with open(output_dir / "sanity_summary.json", "w", encoding="utf-8") as fh:
        json.dump(sanity_summary, fh, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Run dir:        {output_dir}")
    print(f"Sentence files: {sentences_dir}")
    print(f"Warnings:       {len(warnings)}")
    for warning in warnings:
        print(f" - {warning}")


if __name__ == "__main__":
    main()
