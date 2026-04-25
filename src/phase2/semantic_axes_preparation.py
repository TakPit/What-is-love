#!/usr/bin/env python3
"""Build semantic-axis inventory coverage tables against aligned embeddings.

This Phase 2 preparation stage reads an external semantic-axis config JSON and
writes only the two tables needed by the next stage:

  - configX.coverage_by_label.csv
  - configX.item_summary.csv

Outputs are always written under a folder named after the config number:
  <output-root>/configX/

The script can derive the aligned embedding root and ordered labels directly
from the Phase 1 handoff file `phase2_ready.json`, which keeps Phase 2 aligned
with the current Phase 1 repo structure.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

TEXT_EMBEDDING_EXTS = (
    ".txt",
    ".txt.gz",
    ".vec",
    ".vec.gz",
    ".w2v",
    ".w2v.gz",
)

WORD_COLUMNS = ["word", "token", "term", "item"]
COUNT_COLUMNS = ["count", "freq", "frequency", "n"]
DEFAULT_LABELS = ["1970s", "1980s", "1990s", "2000s", "2010s"]


@dataclass
class CoverageContext:
    labels: list[str]
    min_count: int
    allow_target_anchor_overlap: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Semantic-axis config JSON.")
    parser.add_argument(
        "--phase1-ready-json",
        type=Path,
        default=None,
        help="Optional Phase 1 handoff JSON. If provided, aligned-root and labels can be derived from it.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/phase2/semantic_axes_preparation"),
        help="Root output directory. A configX subfolder will be created inside it.",
    )
    parser.add_argument(
        "--aligned-root",
        type=Path,
        default=None,
        help="Aligned embedding run root. Optional if --phase1-ready-json is given.",
    )
    parser.add_argument(
        "--freq-root",
        type=Path,
        default=None,
        help="Optional root for vocab/count CSVs. Defaults to the aligned root.",
    )
    parser.add_argument(
        "--embedding-template",
        default="{label}/model/{label}.kv",
        help="Relative template under --aligned-root for aligned embeddings.",
    )
    parser.add_argument(
        "--freq-template",
        default="{label}/reports/{label}.vocab.csv",
        help="Relative template under --freq-root for per-label vocab/count tables.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=None,
        help="Optional override for settings.min_count in the config.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_word(word: str) -> str:
    return str(word).strip().lower()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def config_label_from_path(path: Path) -> str:
    stem = path.stem
    m = re.search(r"config(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return f"config{m.group(1)}"
    digit_groups = re.findall(r"(\d+)", stem)
    if digit_groups:
        return f"config{digit_groups[-1]}"
    raise ValueError(f"Could not derive config label from filename: {path.name}")


def load_config(path: Path) -> dict[str, Any]:
    cfg = load_json(path)
    require(isinstance(cfg, dict), "Config must be a JSON object.")
    require("targets" in cfg and isinstance(cfg["targets"], list), "Config must contain a 'targets' list.")
    require("axes" in cfg and isinstance(cfg["axes"], list), "Config must contain an 'axes' list.")
    settings = cfg.get("settings", {})
    require(isinstance(settings, dict), "If present, 'settings' must be an object.")
    return cfg


def load_phase1_ready(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_json(path)
    require(isinstance(payload, dict), "phase1_ready_json must contain a JSON object.")
    return payload


def validate_axes(cfg: dict[str, Any]) -> None:
    seen_ids: set[str] = set()
    for axis in cfg["axes"]:
        require(isinstance(axis, dict), "Each axis must be an object.")
        needed = {
            "axis_id",
            "label",
            "positive_pole_name",
            "negative_pole_name",
            "positive_anchors",
            "negative_anchors",
        }
        missing = needed - set(axis.keys())
        require(not missing, f"Axis is missing fields: {sorted(missing)}")
        axis_id = str(axis["axis_id"])
        require(axis_id not in seen_ids, f"Duplicate axis_id: {axis_id}")
        seen_ids.add(axis_id)
        for field in ["positive_anchors", "negative_anchors"]:
            value = axis[field]
            require(isinstance(value, list) and value, f"Axis {axis_id} field '{field}' must be a non-empty list.")


def resolve_aligned_root(args: argparse.Namespace, phase1_ready: dict[str, Any]) -> Path:
    if args.aligned_root is not None:
        return args.aligned_root
    ready_root = phase1_ready.get("aligned_run_dir")
    require(ready_root is not None, "Provide --aligned-root or --phase1-ready-json with aligned_run_dir.")
    return Path(str(ready_root))


def load_settings(cfg: dict[str, Any], cli_min_count: int | None, phase1_ready: dict[str, Any]) -> CoverageContext:
    settings = cfg.get("settings", {})
    labels = settings.get("labels") or settings.get("decades") or phase1_ready.get("ordered_labels") or DEFAULT_LABELS
    require(isinstance(labels, list) and labels, "settings.labels (or settings.decades) must be a non-empty list.")
    min_count = cli_min_count if cli_min_count is not None else int(settings.get("min_count", 20))
    return CoverageContext(
        labels=[str(x) for x in labels],
        min_count=int(min_count),
        allow_target_anchor_overlap=bool(settings.get("allow_target_anchor_overlap", False)),
    )


def choose_existing_path(root: Path, template: str, label: str) -> Path | None:
    path = root / template.format(label=label, decade=label)
    return path if path.exists() else None


def _load_npz_vocab(path: Path) -> set[str]:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    if "words" not in data:
        raise ValueError(f"NPZ file must contain a 'words' array: {path}")
    return {str(word) for word in data["words"].tolist()}


def load_embedding_vocab(path: Path) -> set[str]:
    path_str = str(path)
    if path_str.endswith((".kv", ".kv.gz")):
        kv = KeyedVectors.load(path_str, mmap="r")
        return set(kv.key_to_index.keys())
    if path_str.endswith(".model"):
        model = Word2Vec.load(path_str)
        return set(model.wv.key_to_index.keys())
    if path_str.endswith(".bin"):
        kv = KeyedVectors.load_word2vec_format(path_str, binary=True)
        return set(kv.key_to_index.keys())
    if path_str.endswith(TEXT_EMBEDDING_EXTS):
        kv = KeyedVectors.load_word2vec_format(path_str, binary=False)
        return set(kv.key_to_index.keys())
    if path_str.endswith(".npz"):
        return _load_npz_vocab(path)
    raise ValueError(f"Unsupported embedding format: {path}")


def detect_column(columns: list[str], candidates: list[str], fallback_index: int) -> str:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    if len(columns) > fallback_index:
        return columns[fallback_index]
    raise ValueError(f"Could not detect a suitable column from: {columns}")


def load_frequency_table(path: Path) -> dict[str, int]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    word_col = detect_column(df.columns.tolist(), WORD_COLUMNS, 0)
    count_col = detect_column(df.columns.tolist(), COUNT_COLUMNS, 1)
    words = df[word_col].astype(str).str.strip().str.lower()
    counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)
    return dict(zip(words.tolist(), counts.tolist(), strict=False))


def build_items(cfg: dict[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for target in cfg["targets"]:
        records.append({"inventory_role": "target", "axis_id": "", "pole": "", "word": normalize_word(target)})
    for axis in cfg["axes"]:
        axis_id = str(axis["axis_id"])
        for word in axis["positive_anchors"]:
            records.append({"inventory_role": "anchor", "axis_id": axis_id, "pole": "positive", "word": normalize_word(word)})
        for word in axis["negative_anchors"]:
            records.append({"inventory_role": "anchor", "axis_id": axis_id, "pole": "negative", "word": normalize_word(word)})
    return records


def find_target_anchor_overlap(cfg: dict[str, Any]) -> list[dict[str, str]]:
    targets = {normalize_word(x) for x in cfg.get("targets", [])}
    rows: list[dict[str, str]] = []
    for axis in cfg.get("axes", []):
        axis_id = str(axis["axis_id"])
        for pole, words in (("positive", axis.get("positive_anchors", [])), ("negative", axis.get("negative_anchors", []))):
            for word in words:
                norm = normalize_word(word)
                if norm in targets:
                    rows.append({"axis_id": axis_id, "pole": pole, "word": norm})
    return rows


def safe_write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    phase1_ready = load_phase1_ready(args.phase1_ready_json)
    validate_axes(cfg)
    ctx = load_settings(cfg, args.min_count, phase1_ready)
    cfg_label = config_label_from_path(args.config)
    output_dir = args.output_root / cfg_label
    aligned_root = resolve_aligned_root(args, phase1_ready)
    freq_root = args.freq_root or aligned_root

    overlap_rows = find_target_anchor_overlap(cfg)
    overlap_allowed = ctx.allow_target_anchor_overlap
    overlap_respected = len(overlap_rows) == 0
    if not overlap_allowed and not overlap_respected:
        raise ValueError(
            "Target-anchor overlap is forbidden by default, but overlap was found in the config. Fix the config before running preparation."
        )

    items = build_items(cfg)
    coverage_rows: list[dict[str, Any]] = []

    for label in ctx.labels:
        emb_path = choose_existing_path(aligned_root, args.embedding_template, label)
        freq_path = choose_existing_path(freq_root, args.freq_template, label)
        if emb_path is None:
            raise FileNotFoundError(f"Embedding file not found for label {label}")
        vocab = {normalize_word(word) for word in load_embedding_vocab(emb_path)}
        freqs = load_frequency_table(freq_path) if freq_path else {}

        for item in items:
            present = item["word"] in vocab
            count = freqs.get(item["word"])
            strong = bool(present and count is not None and count >= ctx.min_count)
            coverage_rows.append(
                {
                    "label": label,
                    **item,
                    "present_in_embedding": present,
                    "count": count,
                    f"meets_min_count_{ctx.min_count}": (count >= ctx.min_count) if count is not None else False,
                    "strong_enough": strong,
                }
            )

    coverage_df = pd.DataFrame(coverage_rows)
    grouped = coverage_df.groupby(["inventory_role", "axis_id", "pole", "word"], dropna=False)
    summary_df = grouped.agg(
        labels_present=("present_in_embedding", "sum"),
        labels_strong=("strong_enough", "sum"),
        min_count_observed=("count", lambda s: int(pd.Series(s).dropna().min()) if pd.Series(s).dropna().size else None),
        max_count_observed=("count", lambda s: int(pd.Series(s).dropna().max()) if pd.Series(s).dropna().size else None),
    ).reset_index()
    summary_df["required_labels"] = len(ctx.labels)
    summary_df["passes_all_labels_presence"] = summary_df["labels_present"] == len(ctx.labels)
    summary_df["passes_all_labels_strong"] = summary_df["labels_strong"] == len(ctx.labels)
    summary_df["target_anchor_overlap_allowed"] = overlap_allowed
    summary_df["target_anchor_overlap_respected"] = overlap_respected

    safe_write_csv(output_dir / f"{cfg_label}.coverage_by_label.csv", coverage_df, overwrite=args.overwrite)
    safe_write_csv(output_dir / f"{cfg_label}.item_summary.csv", summary_df, overwrite=args.overwrite)

    print("[ok] Semantic-axis preparation completed.")
    print(f"[ok] Output dir: {output_dir}")
    print(f"[ok] Prefix: {cfg_label}")


if __name__ == "__main__":
    main()
