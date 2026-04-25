#!/usr/bin/env python3
"""Compute semantic axes, target scores, summaries, neighbors, pairwise traces, and plots.

This Phase 2 analysis stage consumes the approved coverage output from
semantic_axes_preparation.py and the aligned embeddings from Phase 1. It can
also derive the aligned root and ordered labels directly from `phase2_ready.json`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

DEFAULT_LABELS = ["1970s", "1980s", "1990s", "2000s", "2010s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Semantic-axis config JSON.")
    parser.add_argument(
        "--item-summary",
        type=Path,
        default=None,
        help="Optional item_summary CSV. If omitted, it is inferred from the config path and label.",
    )
    parser.add_argument(
        "--phase1-ready-json",
        type=Path,
        default=None,
        help="Optional Phase 1 handoff JSON. If provided, aligned-root and ordered labels can be derived from it.",
    )
    parser.add_argument(
        "--aligned-root",
        type=Path,
        default=None,
        help="Aligned embedding root. Optional if --phase1-ready-json is given.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/phase2/semantic_axes"),
        help="Root directory under which a configX subfolder is created.",
    )
    parser.add_argument(
        "--embedding-template",
        default="{label}/model/{label}.kv",
        help="Relative template under --aligned-root for aligned embeddings.",
    )
    parser.add_argument("--neighbors-k", type=int, default=15, help="Number of neighbors saved per target and label.")
    parser.add_argument(
        "--pair-mode",
        choices=["targets_only", "custom_only", "targets_plus_custom"],
        default="targets_only",
        help="Which word pairs to score through time.",
    )
    parser.add_argument(
        "--pair-words-file",
        type=Path,
        default=None,
        help="Optional CSV/TSV with columns word1, word2 for custom pair trajectories.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_word(word: str) -> str:
    return str(word).strip().lower()


def extract_first_year(label: str) -> int:
    digits = re.findall(r"(\d{4})", str(label))
    if digits:
        return int(digits[0])
    raise ValueError(f"Could not extract a 4-digit year from label: {label}")


def config_label_from_path(path: Path) -> str:
    stem = path.stem
    m = re.search(r"config(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return f"config{m.group(1)}"
    digit_groups = re.findall(r"(\d+)", stem)
    if digit_groups:
        return f"config{digit_groups[-1]}"
    raise ValueError(f"Could not derive config label from filename: {path.name}")


def infer_item_summary_path(config_path: Path, cfg_label: str) -> Path:
    return config_path.parent / cfg_label / f"{cfg_label}.item_summary.csv"


def load_item_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "inventory_role",
        "axis_id",
        "pole",
        "word",
        "passes_all_labels_presence",
        "passes_all_labels_strong",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"item_summary is missing required columns {sorted(missing)}: {path}")
    for col in ["inventory_role", "axis_id", "pole", "word"]:
        df[col] = df[col].fillna("").astype(str)
    df["word"] = df["word"].map(normalize_word)
    return df


def load_config(path: Path) -> dict[str, Any]:
    cfg = load_json(path)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a JSON object.")
    return cfg


def load_phase1_ready(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("phase1_ready_json must contain a JSON object.")
    return payload


def resolve_aligned_root(args: argparse.Namespace, phase1_ready: dict[str, Any]) -> Path:
    if args.aligned_root is not None:
        return args.aligned_root
    ready_root = phase1_ready.get("aligned_run_dir")
    if ready_root is None:
        raise ValueError("Provide --aligned-root or --phase1-ready-json with aligned_run_dir.")
    return Path(str(ready_root))


def load_settings(cfg: dict[str, Any], phase1_ready: dict[str, Any]) -> tuple[list[str], int]:
    settings = cfg.get("settings", {}) if isinstance(cfg.get("settings", {}), dict) else {}
    labels = settings.get("labels") or settings.get("decades") or phase1_ready.get("ordered_labels") or DEFAULT_LABELS
    min_count = int(settings.get("min_count", 20))
    return [str(x) for x in labels], min_count


def validate_target_anchor_policy(cfg: dict[str, Any], item_df: pd.DataFrame) -> None:
    respected = True
    if "target_anchor_overlap_respected" in item_df.columns:
        respected = bool(item_df["target_anchor_overlap_respected"].fillna(True).all())
    else:
        targets = {normalize_word(x) for x in cfg.get("targets", [])}
        anchors = set()
        for axis in cfg.get("axes", []):
            anchors.update(normalize_word(x) for x in axis.get("positive_anchors", []))
            anchors.update(normalize_word(x) for x in axis.get("negative_anchors", []))
        respected = len(targets & anchors) == 0
    if not respected:
        raise ValueError("Target-anchor overlap was not respected. Fix the semantic-axis config before running the axis stage.")


def build_pruned_inventory(cfg: dict[str, Any], item_df: pd.DataFrame, min_anchors_per_pole: int = 2) -> dict[str, Any]:
    validate_target_anchor_policy(cfg, item_df)

    item_df = item_df.copy()
    item_df["approved_for_axis_stage"] = item_df["passes_all_labels_presence"] & item_df["passes_all_labels_strong"]

    target_rows = item_df[item_df["inventory_role"] == "target"]
    failed_targets = sorted(set(target_rows.loc[~target_rows["approved_for_axis_stage"], "word"].tolist()))
    if failed_targets:
        raise ValueError(
            "Some targets did not pass approved coverage criteria and should not be scored: " + ", ".join(failed_targets)
        )

    pruned_axes: list[dict[str, Any]] = []
    for axis in cfg.get("axes", []):
        axis_id = str(axis["axis_id"]).strip()
        axis_rows = item_df[(item_df["inventory_role"] == "anchor") & (item_df["axis_id"] == axis_id)]
        pos = axis_rows[(axis_rows["pole"] == "positive") & (axis_rows["approved_for_axis_stage"])]
        neg = axis_rows[(axis_rows["pole"] == "negative") & (axis_rows["approved_for_axis_stage"])]
        pos_words = pos["word"].tolist()
        neg_words = neg["word"].tolist()
        if len(pos_words) < min_anchors_per_pole or len(neg_words) < min_anchors_per_pole:
            continue
        pruned_axis = dict(axis)
        pruned_axis["positive_anchors"] = pos_words
        pruned_axis["negative_anchors"] = neg_words
        pruned_axes.append(pruned_axis)

    if not pruned_axes:
        raise ValueError("No axis remains usable after pruning.")

    return {
        "targets": [normalize_word(x) for x in cfg.get("targets", [])],
        "axes": pruned_axes,
    }


def load_kv(path: Path) -> KeyedVectors:
    return KeyedVectors.load(str(path), mmap="r")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(vec1, vec2) / denom)


def compute_axes_and_scores(
    pruned_inventory: dict[str, Any],
    labels: list[str],
    aligned_root: Path,
    embedding_template: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = [normalize_word(x) for x in pruned_inventory.get("targets", [])]

    axes_by_label: dict[str, Any] = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "created_by_script": "src/phase2/run_semantic_axes.py",
            "labels": labels,
        },
        "axes": {},
    }
    score_rows: list[dict[str, Any]] = []

    for label in labels:
        kv_path = aligned_root / embedding_template.format(label=label, decade=label)
        if not kv_path.exists():
            raise FileNotFoundError(f"Missing aligned embedding file for {label}: {kv_path}")
        kv = load_kv(kv_path)

        for axis in pruned_inventory.get("axes", []):
            axis_id = axis["axis_id"]
            pos_words = [w for w in axis["positive_anchors"] if w in kv.key_to_index]
            neg_words = [w for w in axis["negative_anchors"] if w in kv.key_to_index]
            if len(pos_words) < 2 or len(neg_words) < 2:
                raise ValueError(f"Axis {axis_id} has fewer than 2 usable anchors in {label}.")

            pos_centroid = np.mean(np.vstack([kv[w] for w in pos_words]), axis=0)
            neg_centroid = np.mean(np.vstack([kv[w] for w in neg_words]), axis=0)
            axis_vec = pos_centroid - neg_centroid

            axes_by_label["axes"].setdefault(
                axis_id,
                {
                    "label": axis.get("label", axis_id),
                    "positive_pole_name": axis.get("positive_pole_name", "positive"),
                    "negative_pole_name": axis.get("negative_pole_name", "negative"),
                    "notes": axis.get("notes", ""),
                    "labels": {},
                },
            )
            axes_by_label["axes"][axis_id]["labels"][label] = {
                "positive_anchors_used": pos_words,
                "negative_anchors_used": neg_words,
                "positive_centroid_norm": float(np.linalg.norm(pos_centroid)),
                "negative_centroid_norm": float(np.linalg.norm(neg_centroid)),
                "axis_norm": float(np.linalg.norm(axis_vec)),
                "axis_vector": axis_vec.astype(float).tolist(),
            }

            for target in targets:
                if target not in kv.key_to_index:
                    raise ValueError(f"Target '{target}' is missing from the actual {label} vocabulary.")
                score_rows.append(
                    {
                        "label": label,
                        "axis_id": axis_id,
                        "axis_label": axis.get("label", axis_id),
                        "target": target,
                        "score": cosine_similarity(kv[target], axis_vec),
                        "axis_norm": float(np.linalg.norm(axis_vec)),
                        "positive_anchors_used_n": len(pos_words),
                        "negative_anchors_used_n": len(neg_words),
                    }
                )

    scores_df = pd.DataFrame(score_rows)
    scores_df["_order"] = scores_df["label"].map({label: idx for idx, label in enumerate(labels)})
    scores_df = scores_df.sort_values(["axis_id", "target", "_order"]).drop(columns=["_order"])

    field_summary = (
        scores_df.groupby(["axis_id", "axis_label", "label"], as_index=False)
        .agg(
            target_count=("target", "nunique"),
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
    )
    field_summary["_order"] = field_summary["label"].map({label: idx for idx, label in enumerate(labels)})
    field_summary = field_summary.sort_values(["axis_id", "_order"]).drop(columns=["_order"])

    trajectory_rows: list[dict[str, Any]] = []
    for (axis_id, target), group in scores_df.groupby(["axis_id", "target"], as_index=False):
        g = group.copy()
        g["_order"] = g["label"].map({label: idx for idx, label in enumerate(labels)})
        g = g.sort_values(["_order"])
        first = g.iloc[0]
        last = g.iloc[-1]
        trajectory_rows.append(
            {
                "axis_id": axis_id,
                "axis_label": first["axis_label"],
                "target": target,
                "start_label": first["label"],
                "end_label": last["label"],
                "start_score": float(first["score"]),
                "end_score": float(last["score"]),
                "delta_end_minus_start": float(last["score"] - first["score"]),
                "score_range": float(g["score"].max() - g["score"].min()),
                "mean_score": float(g["score"].mean()),
                "std_score": float(g["score"].std(ddof=1)) if len(g) > 1 else 0.0,
            }
        )
    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["axis_id", "target"])
    return axes_by_label, scores_df, field_summary, trajectory_df


def read_custom_pairs(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    required = {"word1", "word2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pair file must contain columns {sorted(required)}: {path}")
    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        w1 = normalize_word(row["word1"])
        w2 = normalize_word(row["word2"])
        if not w1 or not w2 or w1 == w2:
            continue
        pairs.append(tuple(sorted((w1, w2))))
    return sorted(set(pairs))


def resolve_pairs(targets: list[str], custom_pairs: list[tuple[str, str]], pair_mode: str) -> list[tuple[str, str]]:
    target_pairs = sorted(set(tuple(sorted(pair)) for pair in itertools.combinations(sorted(set(targets)), 2)))
    custom_pairs = sorted(set(tuple(sorted(pair)) for pair in custom_pairs))
    if pair_mode == "targets_only":
        return target_pairs
    if pair_mode == "custom_only":
        return custom_pairs
    if pair_mode == "targets_plus_custom":
        return sorted(set(target_pairs) | set(custom_pairs))
    raise ValueError(f"Unsupported pair_mode: {pair_mode}")


def compute_neighbors_and_pairs(
    targets: list[str],
    pairs: list[tuple[str, str]],
    labels: list[str],
    aligned_root: Path,
    embedding_template: str,
    neighbors_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neighbor_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for label in labels:
        kv_path = aligned_root / embedding_template.format(label=label, decade=label)
        if not kv_path.exists():
            raise FileNotFoundError(f"Missing aligned embedding file for {label}: {kv_path}")
        kv = load_kv(kv_path)

        for target in targets:
            if target not in kv.key_to_index:
                raise ValueError(f"Target '{target}' is missing from the actual {label} vocabulary.")
            neighbors = kv.most_similar(target, topn=neighbors_k)
            for rank, (neighbor_word, similarity) in enumerate(neighbors, start=1):
                neighbor_rows.append(
                    {
                        "label": label,
                        "target": target,
                        "neighbor_rank": rank,
                        "neighbor": normalize_word(neighbor_word),
                        "similarity": float(similarity),
                    }
                )

        for word1, word2 in pairs:
            if word1 not in kv.key_to_index or word2 not in kv.key_to_index:
                raise ValueError(f"Pair word missing from actual {label} vocabulary: ({word1}, {word2}).")
            pair_rows.append(
                {
                    "label": label,
                    "word1": word1,
                    "word2": word2,
                    "pair_label": f"{word1}__{word2}",
                    "similarity": cosine_similarity(kv[word1], kv[word2]),
                }
            )

    neighbors_df = pd.DataFrame(neighbor_rows)
    neighbors_df["_order"] = neighbors_df["label"].map({label: idx for idx, label in enumerate(labels)})
    neighbors_df = neighbors_df.sort_values(["target", "_order", "neighbor_rank"]).drop(columns=["_order"])

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df["_order"] = pairs_df["label"].map({label: idx for idx, label in enumerate(labels)})
    pairs_df = pairs_df.sort_values(["word1", "word2", "_order"]).drop(columns=["_order"])

    pair_summary_rows: list[dict[str, Any]] = []
    for (word1, word2), group in pairs_df.groupby(["word1", "word2"], as_index=False):
        g = group.copy()
        g["_order"] = g["label"].map({label: idx for idx, label in enumerate(labels)})
        g = g.sort_values(["_order"])
        first = g.iloc[0]
        last = g.iloc[-1]
        pair_summary_rows.append(
            {
                "word1": word1,
                "word2": word2,
                "pair_label": first["pair_label"],
                "start_label": first["label"],
                "end_label": last["label"],
                "start_similarity": float(first["similarity"]),
                "end_similarity": float(last["similarity"]),
                "delta_end_minus_start": float(last["similarity"] - first["similarity"]),
                "similarity_range": float(g["similarity"].max() - g["similarity"].min()),
                "mean_similarity": float(g["similarity"].mean()),
                "std_similarity": float(g["similarity"].std(ddof=1)) if len(g) > 1 else 0.0,
            }
        )
    pair_summary_df = pd.DataFrame(pair_summary_rows).sort_values(["word1", "word2"])
    return neighbors_df, pairs_df, pair_summary_df


def safe_write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_plots(
    scores_df: pd.DataFrame,
    field_summary_df: pd.DataFrame,
    output_dir: Path,
    cfg_label: str,
    labels: list[str],
    overwrite: bool,
) -> None:
    x_positions = list(range(len(labels)))
    label_order = {label: idx for idx, label in enumerate(labels)}

    for axis_id, group in scores_df.groupby("axis_id"):
        axis_label = group["axis_label"].iloc[0]
        fig, ax = plt.subplots(figsize=(8, 5))

        for target, tg in group.groupby("target"):
            gg = tg.copy()
            gg["_order"] = gg["label"].map(label_order)
            gg = gg.sort_values(["_order"])
            ax.plot(x_positions, gg["score"].tolist(), marker="o", label=target)

        fg = field_summary_df[field_summary_df["axis_id"] == axis_id].copy()
        if not fg.empty:
            fg["_order"] = fg["label"].map(label_order)
            fg = fg.sort_values(["_order"])
            ax.plot(x_positions, fg["mean_score"].tolist(), marker="s", linewidth=2.5, label="field_mean")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Label")
        ax.set_ylabel("Cosine score on axis")
        ax.set_title(axis_label)
        ax.axhline(0.0, linewidth=1.0)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        out_path = output_dir / f"{cfg_label}.{axis_id}.png"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing plot: {out_path}")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    phase1_ready = load_phase1_ready(args.phase1_ready_json)
    cfg_label = config_label_from_path(args.config)
    item_summary_path = args.item_summary or infer_item_summary_path(args.config, cfg_label)
    item_df = load_item_summary(item_summary_path)
    labels, min_count = load_settings(cfg, phase1_ready)
    aligned_root = resolve_aligned_root(args, phase1_ready)

    pruned_inventory = build_pruned_inventory(cfg, item_df, min_anchors_per_pole=2)
    axes_by_label, scores_df, field_summary_df, trajectory_df = compute_axes_and_scores(
        pruned_inventory=pruned_inventory,
        labels=labels,
        aligned_root=aligned_root,
        embedding_template=args.embedding_template,
    )

    custom_pairs = read_custom_pairs(args.pair_words_file)
    pairs = resolve_pairs(pruned_inventory["targets"], custom_pairs, args.pair_mode)
    neighbors_df, pairwise_df, pair_summary_df = compute_neighbors_and_pairs(
        targets=pruned_inventory["targets"],
        pairs=pairs,
        labels=labels,
        aligned_root=aligned_root,
        embedding_template=args.embedding_template,
        neighbors_k=args.neighbors_k,
    )

    output_dir = args.output_root / cfg_label
    run_summary = {
        "run_name": cfg_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by_script": "src/phase2/run_semantic_axes.py",
        "config": str(args.config),
        "item_summary": str(item_summary_path),
        "phase1_ready_json": str(args.phase1_ready_json) if args.phase1_ready_json else None,
        "aligned_root": str(aligned_root),
        "parameters": {
            "min_count": min_count,
            "neighbors_k": args.neighbors_k,
            "pair_mode": args.pair_mode,
            "custom_pair_file": str(args.pair_words_file) if args.pair_words_file else None,
        },
        "targets": pruned_inventory["targets"],
        "axes": pruned_inventory["axes"],
        "pairs": [{"word1": w1, "word2": w2} for w1, w2 in pairs],
        "labels": labels,
    }

    safe_write_text(output_dir / f"{cfg_label}.run.json", json.dumps(run_summary, indent=2) + "\n", overwrite=args.overwrite)
    safe_write_text(output_dir / f"{cfg_label}.axes_by_label.json", json.dumps(axes_by_label, indent=2) + "\n", overwrite=args.overwrite)
    safe_write_csv(output_dir / f"{cfg_label}.target_axis_scores.csv", scores_df, overwrite=args.overwrite)
    safe_write_csv(output_dir / f"{cfg_label}.field_summary.csv", field_summary_df, overwrite=args.overwrite)
    safe_write_csv(output_dir / f"{cfg_label}.target_trajectory_summary.csv", trajectory_df, overwrite=args.overwrite)
    safe_write_csv(output_dir / f"{cfg_label}.target_neighbors.csv", neighbors_df, overwrite=args.overwrite)
    safe_write_csv(output_dir / f"{cfg_label}.pairwise_similarity_timeseries.csv", pairwise_df, overwrite=args.overwrite)
    safe_write_csv(output_dir / f"{cfg_label}.pairwise_similarity_summary.csv", pair_summary_df, overwrite=args.overwrite)
    make_plots(scores_df, field_summary_df, output_dir, cfg_label, labels, overwrite=args.overwrite)

    print("[ok] Semantic-axis run completed.")
    print(f"[ok] Output dir: {output_dir}")
    print(f"[ok] Prefix: {cfg_label}")
    print(f"[ok] neighbors_k={args.neighbors_k}")
    print(f"[ok] pair_mode={args.pair_mode} | pairs={len(pairs)}")


if __name__ == "__main__":
    main()
