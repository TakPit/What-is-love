#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr

DEFAULT_TARGETS = "love,heart,baby,kiss,touch,forever,mine,desire,hurt,broken"

CLEAN_TOKEN_RE = re.compile(r"^[a-z][a-z'\-]*[a-z]$")
REPEATED_CHAR_RE = re.compile(r"(.)\1\1+")
VOWEL_ELONGATION_RE = re.compile(r"[aeiouy]{3,}")

COMMON_NAME_TOKENS = {
    "abby", "adam", "alex", "alice", "amy", "ana", "andrea", "andy", "angie", "annie",
    "anthony", "arthur", "barbara", "benny", "billy", "bobby", "bonnie", "brian", "charles",
    "charlie", "chris", "christine", "cindy", "claire", "danny", "david", "debbie", "diana",
    "donna", "eddie", "eric", "frank", "gary", "georgia", "gina", "harry", "ian", "jack",
    "jackie", "james", "jane", "janie", "jenny", "jeremy", "jerry", "jim", "jimmy", "joe",
    "john", "johnny", "jon", "jonny", "jose", "judy", "julie", "karen", "kevin", "kim",
    "linda", "lisa", "louie", "lucy", "maggie", "maria", "marvin", "mary", "michael", "mike",
    "nancy", "nick", "nina", "paul", "peter", "rachel", "richard", "rob", "robert", "roger",
    "rose", "sally", "sam", "sandra", "sarah", "scotty", "sharon", "shelly", "simon", "sue",
    "susan", "tammy", "terry", "tim", "tom", "tommy", "tony", "tracy", "vicky", "willy",
}

LIKELY_FILLER_TOKENS = {
    "ah", "aha", "ayy", "eh", "hee", "hey", "hmm", "mah", "mmh", "na", "nah", "oh", "ooh",
    "uh", "uhh", "woah", "yeah", "yea", "yeh", "yo", "yoo", "la", "lalala", "doo", "ohoh",
}


@dataclass
class AlignmentPairSummary:
    source_label: str
    target_label: str
    eligible_shared_words: int
    fit_anchor_words: int
    eval_words: int
    normalize_for_op: bool
    objective_space: Dict[str, float]
    fit_eval: Dict[str, object]
    heldout_eval: Dict[str, object]
    files: Dict[str, str]


@dataclass
class ComparisonSummary:
    comparison_name: str
    left_label: str
    right_label: str
    n_words_full: int
    n_words_clean: int
    distance_stats_full: Dict[str, float]
    distance_stats_clean: Dict[str, float]
    distance_vs_log_min_count_spearman_full: Dict[str, Optional[float]]
    distance_vs_log_min_count_spearman_clean: Dict[str, Optional[float]]
    distance_vs_log_mean_count_spearman_full: Dict[str, Optional[float]]
    distance_vs_log_mean_count_spearman_clean: Dict[str, Optional[float]]
    distance_vs_count_ratio_spearman_full: Dict[str, Optional[float]]
    distance_vs_count_ratio_spearman_clean: Dict[str, Optional[float]]
    files: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chained Orthogonal Procrustes alignment and semantic displacement in one "
            "Phase 1 entry point while keeping alignment and displacement outputs separate."
        )
    )
    parser.add_argument("--input-run-dir", type=str, required=True,
                        help="Training run directory produced by src/phase1/train_sgns_decades.py")
    parser.add_argument("--alignment-output-dir", type=str, required=True,
                        help="Root directory where aligned vectors and alignment diagnostics will be written")
    parser.add_argument("--displacement-output-dir", type=str, required=True,
                        help="Root directory where displacement tables and diagnostics will be written")
    parser.add_argument("--bin-labels", type=str, default="",
                        help="Optional comma-separated ordered bin labels. If omitted, discover from the training run")
    parser.add_argument("--targets", type=str, default=DEFAULT_TARGETS,
                        help="Comma-separated target words used in diagnostics")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into existing non-empty output directories")

    # Alignment options.
    parser.add_argument("--anchor-min-count", type=int, default=100)
    parser.add_argument("--max-anchor-words", type=int, default=20000)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--min-eval-words", type=int, default=500)
    parser.add_argument("--max-eval-words", type=int, default=2000)
    parser.add_argument("--normalize-for-op", action="store_true")
    parser.add_argument("--target-neighbors-topn", type=int, default=20)

    # Displacement options.
    parser.add_argument("--pair-min-count", type=int, default=100)
    parser.add_argument("--stable-min-count", type=int, default=100)
    parser.add_argument("--neighbors-topn", type=int, default=20)
    parser.add_argument("--inspect-top-k-per-pair", type=int, default=50)
    parser.add_argument("--inspect-top-k-range", type=int, default=100)
    return parser.parse_args()


def parse_targets(value: str) -> List[str]:
    return sorted({t.strip().lower() for t in value.split(",") if t.strip()})


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory exists and is not empty: {path}\n"
            "Use --overwrite or choose a different output directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def save_csv(rows: Sequence[Dict[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_label_from_manifest_value(value: str, input_run_dir: Path) -> Optional[str]:
    value = str(value).strip()
    if not value:
        return None
    candidate = input_run_dir / value / "model" / f"{value}.kv"
    if candidate.exists():
        return value
    if re.fullmatch(r"\d{4}", value):
        alt = f"{value}s"
        candidate = input_run_dir / alt / "model" / f"{alt}.kv"
        if candidate.exists():
            return alt
    return None


def discover_bin_labels(input_run_dir: Path, requested: str = "") -> List[str]:
    if requested.strip():
        labels = [x.strip() for x in requested.split(",") if x.strip()]
        if not labels:
            raise ValueError("No labels parsed from --bin-labels")
        return labels

    manifest_path = input_run_dir / "training_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            key = None
            for candidate in ["bin_label", "label", "decade"]:
                if candidate in fieldnames:
                    key = candidate
                    break
            if key is not None:
                labels: List[str] = []
                for row in reader:
                    inferred = infer_label_from_manifest_value(str(row.get(key, "")), input_run_dir)
                    if inferred:
                        labels.append(inferred)
                if labels:
                    return labels

    config_path = input_run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        for key in ["selected_bins", "selected_labels", "selected_decades"]:
            if key in config and isinstance(config[key], list) and config[key]:
                labels = []
                for item in config[key]:
                    inferred = infer_label_from_manifest_value(str(item), input_run_dir)
                    if inferred:
                        labels.append(inferred)
                if labels:
                    return labels

    labels = []
    for child in sorted(input_run_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if not child.is_dir():
            continue
        kv_path = child / "model" / f"{child.name}.kv"
        if kv_path.exists():
            labels.append(child.name)
    if not labels:
        raise FileNotFoundError(
            f"Could not discover any bin-label subdirectories under training run dir: {input_run_dir}"
        )
    return labels


def keyed_vectors_path(run_dir: Path, label: str) -> Path:
    return run_dir / label / "model" / f"{label}.kv"


def load_keyed_vectors(path: Path) -> KeyedVectors:
    if not path.exists():
        raise FileNotFoundError(f"Missing keyed vectors file: {path}")
    return KeyedVectors.load(str(path), mmap="r")


def get_count(kv: KeyedVectors, token: str) -> int:
    try:
        return int(kv.get_vecattr(token, "count"))
    except Exception:
        return 0


def save_vocab_counts_csv(kv: KeyedVectors, output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "token", "count"])
        writer.writeheader()
        for rank, token in enumerate(kv.index_to_key, start=1):
            writer.writerow({"rank": rank, "token": token, "count": get_count(kv, token)})


def save_target_coverage_json(kv: KeyedVectors, targets: Sequence[str], output_path: Path) -> None:
    payload = {}
    for token in targets:
        payload[token] = {"present": token in kv, "count": get_count(kv, token) if token in kv else 0}
    save_json(payload, output_path)


def ranked_eligible_shared_words(left_kv: KeyedVectors, right_kv: KeyedVectors,
                                 anchor_min_count: int, max_anchor_words: int) -> List[str]:
    shared = set(left_kv.index_to_key).intersection(right_kv.index_to_key)
    ranked: List[Tuple[int, int, str]] = []
    for token in shared:
        left_count = get_count(left_kv, token)
        right_count = get_count(right_kv, token)
        if left_count < anchor_min_count or right_count < anchor_min_count:
            continue
        ranked.append((min(left_count, right_count), left_count + right_count, token))
    if not ranked:
        raise ValueError("No shared words survived the anchor_min_count filter.")
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    words = [token for _, _, token in ranked]
    if max_anchor_words > 0:
        words = words[:max_anchor_words]
    return words


def evenly_spaced_indices(n_items: int, n_select: int) -> List[int]:
    if n_select <= 0:
        return []
    if n_select >= n_items:
        return list(range(n_items))
    raw = np.linspace(0, n_items - 1, num=n_select)
    idx = sorted({int(round(x)) for x in raw})
    while len(idx) < n_select:
        for i in range(n_items):
            if i not in idx:
                idx.append(i)
                if len(idx) == n_select:
                    break
    return sorted(idx[:n_select])


def split_fit_and_eval_words(ranked_words: List[str], eval_fraction: float,
                             min_eval_words: int, max_eval_words: int) -> Tuple[List[str], List[str]]:
    n_total = len(ranked_words)
    if n_total < 500:
        raise ValueError(f"Only {n_total} eligible anchor words available. Too small for robust diagnostics.")
    desired_eval = int(round(n_total * eval_fraction))
    desired_eval = max(desired_eval, min_eval_words)
    desired_eval = min(desired_eval, max_eval_words)
    desired_eval = min(desired_eval, max(0, n_total - 200))
    eval_idx = set(evenly_spaced_indices(n_total, desired_eval))
    eval_words = [word for i, word in enumerate(ranked_words) if i in eval_idx]
    fit_words = [word for i, word in enumerate(ranked_words) if i not in eval_idx]
    if len(fit_words) < 200:
        raise ValueError(f"Only {len(fit_words)} fit anchor words remain after hold-out split.")
    if len(eval_words) < 100:
        raise ValueError(f"Only {len(eval_words)} held-out eval words remain after split.")
    return fit_words, eval_words


def save_word_list_csv(words: Sequence[str], left_kv: KeyedVectors, right_kv: KeyedVectors, output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rank", "token", "left_count", "right_count", "min_count", "sum_count"],
        )
        writer.writeheader()
        for rank, token in enumerate(words, start=1):
            left_count = get_count(left_kv, token)
            right_count = get_count(right_kv, token)
            writer.writerow({
                "rank": rank,
                "token": token,
                "left_count": left_count,
                "right_count": right_count,
                "min_count": min(left_count, right_count),
                "sum_count": left_count + right_count,
            })


def build_matrix(kv: KeyedVectors, words: Sequence[str], normalize: bool) -> np.ndarray:
    mat = np.vstack([kv[word] for word in words]).astype(np.float64)
    if normalize:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat = mat / norms
    return mat


def summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def cosine_alignment_table(source_kv: KeyedVectors, target_kv: KeyedVectors,
                           words: Sequence[str], rotation: np.ndarray) -> List[Dict[str, object]]:
    source = build_matrix(source_kv, words, normalize=True)
    target = build_matrix(target_kv, words, normalize=True)
    before = np.sum(source * target, axis=1)
    after = np.sum((source @ rotation) * target, axis=1)
    delta = after - before
    rows: List[Dict[str, object]] = []
    for i, token in enumerate(words):
        rows.append({
            "token": token,
            "source_count": get_count(source_kv, token),
            "target_count": get_count(target_kv, token),
            "cosine_before": float(before[i]),
            "cosine_after": float(after[i]),
            "delta": float(delta[i]),
        })
    return rows


def save_cosine_table_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["token", "source_count", "target_count", "cosine_before", "cosine_after", "delta"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def diagnostics_from_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    before = np.array([float(r["cosine_before"]) for r in rows], dtype=np.float64)
    after = np.array([float(r["cosine_after"]) for r in rows], dtype=np.float64)
    delta = after - before
    improved = delta > 0
    worsened = delta < 0
    return {
        "n_words": len(rows),
        "cosine_before": summary_stats(before),
        "cosine_after": summary_stats(after),
        "delta": summary_stats(delta),
        "fraction_improved": float(np.mean(improved)),
        "fraction_worsened": float(np.mean(worsened)),
        "fraction_unchanged": float(np.mean(delta == 0)),
        "mean_gain": float(np.mean(delta)),
        "median_gain": float(np.median(delta)),
    }


def top_rows(rows: Sequence[Dict[str, object]], key: str, n: int, reverse: bool = True) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda r: float(r[key]), reverse=reverse)[:n]


def fit_procrustes_rotation(source_kv: KeyedVectors, target_kv: KeyedVectors,
                            fit_words: Sequence[str], normalize_for_op: bool) -> Tuple[np.ndarray, Dict[str, float]]:
    source_fit = build_matrix(source_kv, fit_words, normalize=normalize_for_op)
    target_fit = build_matrix(target_kv, fit_words, normalize=normalize_for_op)
    fro_before = float(np.linalg.norm(source_fit - target_fit, ord="fro"))
    rotation, _ = orthogonal_procrustes(source_fit, target_fit)
    source_fit_aligned = source_fit @ rotation
    fro_after = float(np.linalg.norm(source_fit_aligned - target_fit, ord="fro"))
    ortho_err = float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]), ord="fro"))
    det = float(np.linalg.det(rotation))
    return rotation, {
        "frobenius_before_objective_space": fro_before,
        "frobenius_after_objective_space": fro_after,
        "frobenius_relative_reduction_objective_space": ((fro_before - fro_after) / fro_before if fro_before > 0 else 0.0),
        "rotation_orthogonality_error_fro": ortho_err,
        "rotation_determinant": det,
    }


def create_aligned_keyedvectors(kv: KeyedVectors, rotation: np.ndarray) -> KeyedVectors:
    rotated = kv.vectors.astype(np.float64) @ rotation
    aligned = KeyedVectors(vector_size=kv.vector_size)
    aligned.add_vectors(kv.index_to_key, rotated.astype(np.float32))
    for token in kv.index_to_key:
        count = get_count(kv, token)
        if count > 0:
            try:
                aligned.set_vecattr(token, "count", int(count))
            except Exception:
                pass
    aligned.fill_norms(force=True)
    return aligned


def target_self_cosine_diagnostics(prev_aligned_kv: KeyedVectors, curr_original_kv: KeyedVectors,
                                   curr_aligned_kv: KeyedVectors, targets: Sequence[str]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for token in targets:
        present_prev = token in prev_aligned_kv
        present_curr = token in curr_original_kv
        if not (present_prev and present_curr):
            payload[token] = {"present_in_prev": present_prev, "present_in_curr": present_curr}
            continue
        v_prev = prev_aligned_kv.get_vector(token, norm=True)
        v_curr_before = curr_original_kv.get_vector(token, norm=True)
        v_curr_after = curr_aligned_kv.get_vector(token, norm=True)
        payload[token] = {
            "present_in_prev": True,
            "present_in_curr": True,
            "prev_count": get_count(prev_aligned_kv, token),
            "curr_count": get_count(curr_original_kv, token),
            "self_cosine_before": float(np.dot(v_prev, v_curr_before)),
            "self_cosine_after": float(np.dot(v_prev, v_curr_after)),
            "delta": float(np.dot(v_prev, v_curr_after) - np.dot(v_prev, v_curr_before)),
        }
    return payload


def target_neighbor_diagnostics(prev_aligned_kv: KeyedVectors, curr_original_kv: KeyedVectors,
                                curr_aligned_kv: KeyedVectors, targets: Sequence[str], topn: int) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for token in targets:
        if token not in prev_aligned_kv or token not in curr_original_kv or token not in curr_aligned_kv:
            payload[token] = {"available": False}
            continue
        prev_neighbors = prev_aligned_kv.most_similar(token, topn=topn)
        curr_before_neighbors = curr_original_kv.most_similar(token, topn=topn)
        curr_after_neighbors = curr_aligned_kv.most_similar(token, topn=topn)
        prev_set = {w for w, _ in prev_neighbors}
        before_set = {w for w, _ in curr_before_neighbors}
        after_set = {w for w, _ in curr_after_neighbors}
        payload[token] = {
            "available": True,
            "prev_neighbors": [[w, float(s)] for w, s in prev_neighbors],
            "curr_before_neighbors": [[w, float(s)] for w, s in curr_before_neighbors],
            "curr_after_neighbors": [[w, float(s)] for w, s in curr_after_neighbors],
            "jaccard_prev_vs_curr_before": float(len(prev_set & before_set) / len(prev_set | before_set) if (prev_set | before_set) else 0.0),
            "jaccard_prev_vs_curr_after": float(len(prev_set & after_set) / len(prev_set | after_set) if (prev_set | after_set) else 0.0),
        }
    return payload


def save_aligned_artifacts(kv: KeyedVectors, label: str, output_dir: Path,
                           cumulative_rotation_to_base: np.ndarray, targets: Sequence[str]) -> Dict[str, str]:
    label_dir = output_dir / label
    model_dir = label_dir / "model"
    report_dir = label_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    kv_path = model_dir / f"{label}.kv"
    txt_path = model_dir / f"{label}.vectors.txt"
    bin_path = model_dir / f"{label}.vectors.bin"
    rot_path = model_dir / f"{label}.cumulative_rotation_to_base.npy"
    vocab_csv = report_dir / f"{label}.vocab.csv"
    coverage_json = report_dir / f"{label}.target_coverage.json"

    kv.save(str(kv_path))
    kv.save_word2vec_format(str(txt_path), binary=False)
    kv.save_word2vec_format(str(bin_path), binary=True)
    np.save(rot_path, cumulative_rotation_to_base.astype(np.float32))
    save_vocab_counts_csv(kv, vocab_csv)
    save_target_coverage_json(kv, list(targets), coverage_json)
    return {
        "keyed_vectors": str(kv_path),
        "vectors_txt": str(txt_path),
        "vectors_bin": str(bin_path),
        "cumulative_rotation_to_base": str(rot_path),
        "vocab_csv": str(vocab_csv),
        "target_coverage": str(coverage_json),
    }


def is_probable_name(token: str) -> bool:
    return token in COMMON_NAME_TOKENS


def is_probable_filler(token: str) -> bool:
    return token in LIKELY_FILLER_TOKENS or bool(VOWEL_ELONGATION_RE.search(token))


def token_quality_flags(token: str) -> List[str]:
    flags: List[str] = []
    if len(token) < 3:
        flags.append("too_short")
    if any(ch.isdigit() for ch in token):
        flags.append("has_digit")
    if not CLEAN_TOKEN_RE.match(token):
        flags.append("non_standard_chars")
    if REPEATED_CHAR_RE.search(token):
        flags.append("elongated_spelling")
    if token.startswith("'") or token.endswith("'"):
        flags.append("edge_apostrophe")
    if token.count("-") > 1:
        flags.append("multiple_hyphens")
    if is_probable_name(token):
        flags.append("probable_name")
    if is_probable_filler(token):
        flags.append("probable_filler")
    return flags


def is_clean_token(token: str) -> bool:
    return len(token_quality_flags(token)) == 0


def shared_words_for_pair(left_kv: KeyedVectors, right_kv: KeyedVectors, min_count: int) -> List[str]:
    shared = set(left_kv.index_to_key).intersection(right_kv.index_to_key)
    words = [token for token in shared if get_count(left_kv, token) >= min_count and get_count(right_kv, token) >= min_count]
    words.sort()
    return words


def shared_words_all_labels(kv_by_label: Dict[str, KeyedVectors], labels: Sequence[str], min_count: int) -> List[str]:
    shared = set(kv_by_label[labels[0]].index_to_key)
    for label in labels[1:]:
        shared &= set(kv_by_label[label].index_to_key)
    words = []
    for token in shared:
        if all(get_count(kv_by_label[label], token) >= min_count for label in labels):
            words.append(token)
    words.sort()
    return words


def normalized_vector(kv: KeyedVectors, token: str) -> np.ndarray:
    return kv.get_vector(token, norm=True)


def cosine_similarity_between(left_kv: KeyedVectors, right_kv: KeyedVectors, token: str) -> float:
    return float(np.dot(normalized_vector(left_kv, token), normalized_vector(right_kv, token)))


def nearest_neighbors(kv: KeyedVectors, token: str, topn: int) -> List[List[object]]:
    return [[word, float(score)] for word, score in kv.most_similar(token, topn=topn)]


def jaccard_of_neighbor_sets(left_kv: KeyedVectors, right_kv: KeyedVectors, token: str, topn: int) -> float:
    left = {word for word, _ in left_kv.most_similar(token, topn=topn)}
    right = {word for word, _ in right_kv.most_similar(token, topn=topn)}
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def safe_ratio(max_value: float, min_value: float) -> float:
    if min_value <= 0:
        return float("inf")
    return float(max_value / min_value)


def coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(arr))
    if mean == 0.0:
        return 0.0
    return float(np.std(arr) / mean)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(x) < 3 or len(y) < 3:
        return {"rho": None, "pvalue": None}
    if len(set(x)) <= 1 or len(set(y)) <= 1:
        return {"rho": None, "pvalue": None}
    rho, pval = spearmanr(x, y)
    if np.isnan(rho) or np.isnan(pval):
        return {"rho": None, "pvalue": None}
    return {"rho": float(rho), "pvalue": float(pval)}


def build_pairwise_rows(left_kv: KeyedVectors, right_kv: KeyedVectors,
                        words: Sequence[str], left_label: str, right_label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for token in words:
        sim = cosine_similarity_between(left_kv, right_kv, token)
        left_count = get_count(left_kv, token)
        right_count = get_count(right_kv, token)
        flags = token_quality_flags(token)
        rows.append({
            "token": token,
            "left_label": left_label,
            "right_label": right_label,
            "cosine_similarity": sim,
            "cosine_distance": float(1.0 - sim),
            "left_count": left_count,
            "right_count": right_count,
            "min_count": min(left_count, right_count),
            "max_count": max(left_count, right_count),
            "mean_count": float((left_count + right_count) / 2.0),
            "count_ratio_max_to_min": safe_ratio(max(left_count, right_count), max(1, min(left_count, right_count))),
            "clean_token": is_clean_token(token),
            "is_short_token": len(token) < 3,
            "has_repeated_char_pattern": bool(REPEATED_CHAR_RE.search(token)),
            "has_nonstandard_chars": not bool(CLEAN_TOKEN_RE.match(token)),
            "is_probable_name": is_probable_name(token),
            "is_probable_filler": is_probable_filler(token),
            "quality_flags": "|".join(flags),
        })
    rows.sort(key=lambda r: (-float(r["cosine_distance"]), str(r["token"])))
    return rows


def add_top_neighbor_diagnostics(kv_by_label: Dict[str, KeyedVectors], diagnostics_tokens: Sequence[str],
                                 labels: Sequence[str], topn: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for token in diagnostics_tokens:
        token_payload: Dict[str, object] = {}
        for label in labels:
            kv = kv_by_label[label]
            if token not in kv:
                token_payload[label] = {"present": False}
                continue
            token_payload[label] = {"present": True, "count": get_count(kv, token), "neighbors": nearest_neighbors(kv, token, topn=topn)}
        for left_label, right_label in zip(labels[:-1], labels[1:]):
            key = f"jaccard_{left_label}__to__{right_label}"
            left_kv = kv_by_label[left_label]
            right_kv = kv_by_label[right_label]
            if token in left_kv and token in right_kv:
                token_payload[key] = jaccard_of_neighbor_sets(left_kv, right_kv, token, topn)
            else:
                token_payload[key] = None
        out[token] = token_payload
    return out


def comparison_summary(comparison_name: str, left_label: str, right_label: str,
                       rows_full: Sequence[Dict[str, object]], rows_clean: Sequence[Dict[str, object]],
                       files: Dict[str, str]) -> ComparisonSummary:
    full_distances = [float(r["cosine_distance"]) for r in rows_full]
    clean_distances = [float(r["cosine_distance"]) for r in rows_clean]
    full_log_min_counts = [math.log(max(1, int(r["min_count"]))) for r in rows_full]
    clean_log_min_counts = [math.log(max(1, int(r["min_count"]))) for r in rows_clean]
    full_log_mean_counts = [math.log(max(1.0, float(r["mean_count"]))) for r in rows_full]
    clean_log_mean_counts = [math.log(max(1.0, float(r["mean_count"]))) for r in rows_clean]
    full_count_ratio = [float(r["count_ratio_max_to_min"]) for r in rows_full]
    clean_count_ratio = [float(r["count_ratio_max_to_min"]) for r in rows_clean]
    return ComparisonSummary(
        comparison_name=comparison_name,
        left_label=left_label,
        right_label=right_label,
        n_words_full=len(rows_full),
        n_words_clean=len(rows_clean),
        distance_stats_full=summary_stats(full_distances),
        distance_stats_clean=summary_stats(clean_distances),
        distance_vs_log_min_count_spearman_full=safe_spearman(full_distances, full_log_min_counts),
        distance_vs_log_min_count_spearman_clean=safe_spearman(clean_distances, clean_log_min_counts),
        distance_vs_log_mean_count_spearman_full=safe_spearman(full_distances, full_log_mean_counts),
        distance_vs_log_mean_count_spearman_clean=safe_spearman(clean_distances, clean_log_mean_counts),
        distance_vs_count_ratio_spearman_full=safe_spearman(full_distances, full_count_ratio),
        distance_vs_count_ratio_spearman_clean=safe_spearman(clean_distances, clean_count_ratio),
        files=files,
    )


def build_stable_rows(kv_by_label: Dict[str, KeyedVectors], labels: Sequence[str], words: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    adjacent_pairs = list(zip(labels[:-1], labels[1:]))
    for token in words:
        counts = {label: get_count(kv_by_label[label], token) for label in labels}
        count_values = list(counts.values())
        similarities = {}
        distances = {}
        adj_distances: List[float] = []
        pair_labels: List[str] = []
        for left_label, right_label in adjacent_pairs:
            sim = cosine_similarity_between(kv_by_label[left_label], kv_by_label[right_label], token)
            dist = float(1.0 - sim)
            similarities[f"cos_{left_label}__to__{right_label}"] = sim
            distances[f"dist_{left_label}__to__{right_label}"] = dist
            adj_distances.append(dist)
            pair_labels.append(f"{left_label}__to__{right_label}")
        range_sim = cosine_similarity_between(kv_by_label[labels[0]], kv_by_label[labels[-1]], token)
        range_dist = float(1.0 - range_sim)
        flags = token_quality_flags(token)
        max_idx = int(np.argmax(adj_distances))
        dominant_pair = pair_labels[max_idx]
        adj_sum = float(np.sum(adj_distances))
        adj_max = float(np.max(adj_distances))
        one_step_dominance = float(adj_max / adj_sum) if adj_sum > 0 else 0.0
        row: Dict[str, object] = {
            "token": token,
            "clean_token": is_clean_token(token),
            "is_short_token": len(token) < 3,
            "has_repeated_char_pattern": bool(REPEATED_CHAR_RE.search(token)),
            "has_nonstandard_chars": not bool(CLEAN_TOKEN_RE.match(token)),
            "is_probable_name": is_probable_name(token),
            "is_probable_filler": is_probable_filler(token),
            "quality_flags": "|".join(flags),
            "min_count_all_labels": min(count_values),
            "max_count_all_labels": max(count_values),
            "mean_count_all_labels": float(np.mean(count_values)),
            "count_ratio_max_to_min": safe_ratio(max(count_values), max(1, min(count_values))),
            "count_cv_all_labels": coefficient_of_variation(count_values),
            "range_cosine_similarity": range_sim,
            "range_cosine_distance": range_dist,
            "adjacent_distance_mean": float(np.mean(adj_distances)),
            "adjacent_distance_sum": adj_sum,
            "adjacent_distance_max": adj_max,
            "dominant_jump_pair": dominant_pair,
            "one_step_dominance": one_step_dominance,
        }
        for label in labels:
            row[f"count_{label}"] = counts[label]
        row.update(similarities)
        row.update(distances)
        rows.append(row)
    rows.sort(key=lambda r: (-float(r["range_cosine_distance"]), str(r["token"])))
    return rows


def target_trajectory_rows(kv_by_label: Dict[str, KeyedVectors], labels: Sequence[str], targets: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    adjacent_pairs = list(zip(labels[:-1], labels[1:]))
    for token in targets:
        present = all(token in kv_by_label[label] for label in labels)
        row: Dict[str, object] = {"token": token, "present_all_labels": present, "is_probable_name": is_probable_name(token), "is_probable_filler": is_probable_filler(token)}
        counts = []
        for label in labels:
            c = get_count(kv_by_label[label], token) if token in kv_by_label[label] else 0
            row[f"count_{label}"] = c
            counts.append(c)
        if present:
            adj_distances = []
            pair_names = []
            for left_label, right_label in adjacent_pairs:
                sim = cosine_similarity_between(kv_by_label[left_label], kv_by_label[right_label], token)
                dist = float(1.0 - sim)
                row[f"cos_{left_label}__to__{right_label}"] = sim
                row[f"dist_{left_label}__to__{right_label}"] = dist
                adj_distances.append(dist)
                pair_names.append(f"{left_label}__to__{right_label}")
            range_sim = cosine_similarity_between(kv_by_label[labels[0]], kv_by_label[labels[-1]], token)
            range_dist = float(1.0 - range_sim)
            row["range_cosine_similarity"] = range_sim
            row["range_cosine_distance"] = range_dist
            row["adjacent_distance_mean"] = float(np.mean(adj_distances))
            row["adjacent_distance_sum"] = float(np.sum(adj_distances))
            row["adjacent_distance_max"] = float(np.max(adj_distances))
            row["count_ratio_max_to_min"] = safe_ratio(max(counts), max(1, min(counts)))
            row["count_cv_all_labels"] = coefficient_of_variation(counts)
            max_idx = int(np.argmax(adj_distances))
            row["dominant_jump_pair"] = pair_names[max_idx]
            total = float(np.sum(adj_distances))
            row["one_step_dominance"] = float(np.max(adj_distances) / total) if total > 0 else 0.0
        rows.append(row)
    return rows


def write_ranked_pair_outputs(output_root: Path, comparison_name: str,
                              rows_full: Sequence[Dict[str, object]], rows_clean: Sequence[Dict[str, object]],
                              inspect_tokens: Sequence[str], neighbor_diag: Dict[str, object],
                              summary_obj: ComparisonSummary) -> None:
    comp_dir = output_root / comparison_name
    comp_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token", "left_label", "right_label", "cosine_similarity", "cosine_distance",
        "left_count", "right_count", "min_count", "max_count", "mean_count",
        "count_ratio_max_to_min", "clean_token", "is_short_token", "has_repeated_char_pattern",
        "has_nonstandard_chars", "is_probable_name", "is_probable_filler", "quality_flags",
    ]
    save_csv(rows_full, comp_dir / f"{comparison_name}.full.csv", fieldnames)
    save_csv(rows_clean, comp_dir / f"{comparison_name}.clean.csv", fieldnames)
    save_csv(list(rows_full)[:100], comp_dir / f"{comparison_name}.top100_full.csv", fieldnames)
    save_csv(list(rows_clean)[:100], comp_dir / f"{comparison_name}.top100_clean.csv", fieldnames)
    save_json({"tokens": list(inspect_tokens), "neighbor_diagnostics": neighbor_diag}, comp_dir / f"{comparison_name}.neighbors.json")
    save_json(asdict(summary_obj), comp_dir / f"{comparison_name}.summary.json")


def write_stable_outputs(output_dir: Path, rows_full: Sequence[Dict[str, object]], rows_clean: Sequence[Dict[str, object]],
                         labels: Sequence[str], inspect_top_k: int, kv_by_label: Dict[str, KeyedVectors],
                         neighbors_topn: int) -> Dict[str, str]:
    stable_dir = output_dir / "stable_all_labels"
    stable_dir.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "token", "clean_token", "is_short_token", "has_repeated_char_pattern", "has_nonstandard_chars",
        "is_probable_name", "is_probable_filler", "quality_flags", "min_count_all_labels",
        "max_count_all_labels", "mean_count_all_labels", "count_ratio_max_to_min", "count_cv_all_labels",
        "range_cosine_similarity", "range_cosine_distance", "adjacent_distance_mean", "adjacent_distance_sum",
        "adjacent_distance_max", "dominant_jump_pair", "one_step_dominance",
    ]
    count_fields = [f"count_{label}" for label in labels]
    sim_fields = [f"cos_{left}__to__{right}" for left, right in zip(labels[:-1], labels[1:])]
    dist_fields = [f"dist_{left}__to__{right}" for left, right in zip(labels[:-1], labels[1:])]
    fieldnames = base_fields + count_fields + sim_fields + dist_fields
    save_csv(rows_full, stable_dir / "stable_all_labels.full.csv", fieldnames)
    save_csv(rows_clean, stable_dir / "stable_all_labels.clean.csv", fieldnames)

    by_range_full = sorted(rows_full, key=lambda r: (-float(r["range_cosine_distance"]), str(r["token"])))
    by_range_clean = sorted(rows_clean, key=lambda r: (-float(r["range_cosine_distance"]), str(r["token"])))
    by_sum_clean = sorted(rows_clean, key=lambda r: (-float(r["adjacent_distance_sum"]), str(r["token"])))
    by_mean_clean = sorted(rows_clean, key=lambda r: (-float(r["adjacent_distance_mean"]), str(r["token"])))
    by_max_clean = sorted(rows_clean, key=lambda r: (-float(r["adjacent_distance_max"]), str(r["token"])))

    save_csv(by_range_full[:200], stable_dir / "top200_by_range_distance.full.csv", fieldnames)
    save_csv(by_range_clean[:200], stable_dir / "top200_by_range_distance.clean.csv", fieldnames)
    save_csv(by_sum_clean[:200], stable_dir / "top200_by_adjacent_sum.clean.csv", fieldnames)
    save_csv(by_mean_clean[:200], stable_dir / "top200_by_adjacent_mean.clean.csv", fieldnames)
    save_csv(by_max_clean[:200], stable_dir / "top200_by_adjacent_max.clean.csv", fieldnames)

    range_distances_full = [float(r["range_cosine_distance"]) for r in rows_full]
    range_distances_clean = [float(r["range_cosine_distance"]) for r in rows_clean]
    log_min_full = [math.log(max(1, int(r["min_count_all_labels"]))) for r in rows_full]
    log_min_clean = [math.log(max(1, int(r["min_count_all_labels"]))) for r in rows_clean]
    log_mean_full = [math.log(max(1.0, float(r["mean_count_all_labels"]))) for r in rows_full]
    log_mean_clean = [math.log(max(1.0, float(r["mean_count_all_labels"]))) for r in rows_clean]
    count_ratio_full = [float(r["count_ratio_max_to_min"]) for r in rows_full]
    count_ratio_clean = [float(r["count_ratio_max_to_min"]) for r in rows_clean]
    one_step_full = [float(r["one_step_dominance"]) for r in rows_full]
    one_step_clean = [float(r["one_step_dominance"]) for r in rows_clean]

    summary_payload = {
        "n_words_full": len(rows_full),
        "n_words_clean": len(rows_clean),
        "range_distance_stats_full": summary_stats(range_distances_full),
        "range_distance_stats_clean": summary_stats(range_distances_clean),
        "range_distance_vs_log_min_count_spearman_full": safe_spearman(range_distances_full, log_min_full),
        "range_distance_vs_log_min_count_spearman_clean": safe_spearman(range_distances_clean, log_min_clean),
        "range_distance_vs_log_mean_count_spearman_full": safe_spearman(range_distances_full, log_mean_full),
        "range_distance_vs_log_mean_count_spearman_clean": safe_spearman(range_distances_clean, log_mean_clean),
        "range_distance_vs_count_ratio_spearman_full": safe_spearman(range_distances_full, count_ratio_full),
        "range_distance_vs_count_ratio_spearman_clean": safe_spearman(range_distances_clean, count_ratio_clean),
        "range_distance_vs_one_step_dominance_spearman_full": safe_spearman(range_distances_full, one_step_full),
        "range_distance_vs_one_step_dominance_spearman_clean": safe_spearman(range_distances_clean, one_step_clean),
    }
    save_json(summary_payload, stable_dir / "stable_all_labels.summary.json")
    inspect_tokens = [str(r["token"]) for r in by_range_clean[:inspect_top_k]]
    save_json(
        add_top_neighbor_diagnostics(kv_by_label=kv_by_label, diagnostics_tokens=inspect_tokens, labels=labels, topn=neighbors_topn),
        stable_dir / "top_range_candidates.neighbors.json",
    )
    return {
        "stable_all_labels_full_csv": str(stable_dir / "stable_all_labels.full.csv"),
        "stable_all_labels_clean_csv": str(stable_dir / "stable_all_labels.clean.csv"),
        "stable_all_labels_summary_json": str(stable_dir / "stable_all_labels.summary.json"),
        "top_range_candidates_neighbors_json": str(stable_dir / "top_range_candidates.neighbors.json"),
    }


def run_alignment(input_run_dir: Path, output_dir: Path, labels: Sequence[str], targets: Sequence[str], args: argparse.Namespace) -> Dict[str, object]:
    logging.info("Starting chained Orthogonal Procrustes alignment")
    save_json(
        {
            "input_run_dir": str(input_run_dir),
            "output_dir": str(output_dir),
            "ordered_labels": list(labels),
            "targets": list(targets),
            "alignment_method": "chained_orthogonal_procrustes",
            "base_label": labels[0],
            "anchor_min_count": args.anchor_min_count,
            "max_anchor_words": args.max_anchor_words,
            "eval_fraction": args.eval_fraction,
            "min_eval_words": args.min_eval_words,
            "max_eval_words": args.max_eval_words,
            "normalize_for_op": args.normalize_for_op,
            "target_neighbors_topn": args.target_neighbors_topn,
        },
        output_dir / "run_config.json",
    )

    overall_start = time.time()
    manifest_rows: List[Dict[str, object]] = []
    pair_dir = output_dir / "pair_reports"
    pair_dir.mkdir(parents=True, exist_ok=True)

    base_label = labels[0]
    base_kv = load_keyed_vectors(keyed_vectors_path(input_run_dir, base_label))
    identity = np.eye(base_kv.vector_size, dtype=np.float64)
    base_outputs = save_aligned_artifacts(base_kv, base_label, output_dir, identity, targets)
    save_json(
        {
            "label": base_label,
            "is_base_label": True,
            "aligned_to_previous_label": None,
            "output_files": base_outputs,
        },
        output_dir / base_label / "reports" / f"{base_label}.alignment_summary.json",
    )
    manifest_rows.append({
        "label": base_label,
        "is_base_label": True,
        "aligned_to_previous_label": "",
        "eligible_shared_words": "",
        "fit_anchor_words": "",
        "eval_words": "",
        "fit_mean_cos_before": "",
        "fit_mean_cos_after": "",
        "heldout_mean_cos_before": "",
        "heldout_mean_cos_after": "",
        "rotation_orthogonality_error_fro": "0.00000000",
        "cumulative_rotation_to_base": base_outputs["cumulative_rotation_to_base"],
        "keyed_vectors": base_outputs["keyed_vectors"],
    })

    prev_aligned_kv = base_kv
    for prev_label, curr_label in zip(labels[:-1], labels[1:]):
        logging.info("Aligning %s -> %s", curr_label, prev_label)
        curr_original_kv = load_keyed_vectors(keyed_vectors_path(input_run_dir, curr_label))
        ranked_words = ranked_eligible_shared_words(prev_aligned_kv, curr_original_kv, args.anchor_min_count, args.max_anchor_words)
        fit_words, eval_words = split_fit_and_eval_words(ranked_words, args.eval_fraction, args.min_eval_words, args.max_eval_words)

        fit_csv = pair_dir / f"{prev_label}_to_{curr_label}.fit_words.csv"
        eval_csv = pair_dir / f"{prev_label}_to_{curr_label}.eval_words.csv"
        save_word_list_csv(fit_words, prev_aligned_kv, curr_original_kv, fit_csv)
        save_word_list_csv(eval_words, prev_aligned_kv, curr_original_kv, eval_csv)

        rotation, objective_space = fit_procrustes_rotation(curr_original_kv, prev_aligned_kv, fit_words, args.normalize_for_op)
        fit_rows = cosine_alignment_table(curr_original_kv, prev_aligned_kv, fit_words, rotation)
        eval_rows = cosine_alignment_table(curr_original_kv, prev_aligned_kv, eval_words, rotation)
        fit_diag = diagnostics_from_rows(fit_rows)
        heldout_diag = diagnostics_from_rows(eval_rows)

        fit_cos_csv = pair_dir / f"{prev_label}_to_{curr_label}.fit_wordwise_cosines.csv"
        eval_cos_csv = pair_dir / f"{prev_label}_to_{curr_label}.heldout_wordwise_cosines.csv"
        save_cosine_table_csv(fit_rows, fit_cos_csv)
        save_cosine_table_csv(eval_rows, eval_cos_csv)

        fit_top_gain_csv = pair_dir / f"{prev_label}_to_{curr_label}.fit_top_gains.csv"
        fit_worst_after_csv = pair_dir / f"{prev_label}_to_{curr_label}.fit_worst_after.csv"
        eval_top_gain_csv = pair_dir / f"{prev_label}_to_{curr_label}.heldout_top_gains.csv"
        eval_worst_after_csv = pair_dir / f"{prev_label}_to_{curr_label}.heldout_worst_after.csv"
        save_cosine_table_csv(top_rows(fit_rows, key="delta", n=100, reverse=True), fit_top_gain_csv)
        save_cosine_table_csv(top_rows(fit_rows, key="cosine_after", n=100, reverse=False), fit_worst_after_csv)
        save_cosine_table_csv(top_rows(eval_rows, key="delta", n=100, reverse=True), eval_top_gain_csv)
        save_cosine_table_csv(top_rows(eval_rows, key="cosine_after", n=100, reverse=False), eval_worst_after_csv)

        curr_aligned_kv = create_aligned_keyedvectors(curr_original_kv, rotation)
        target_self_path = pair_dir / f"{prev_label}_to_{curr_label}.target_self_cosines.json"
        target_neighbors_path = pair_dir / f"{prev_label}_to_{curr_label}.target_neighbors.json"
        save_json(target_self_cosine_diagnostics(prev_aligned_kv, curr_original_kv, curr_aligned_kv, targets), target_self_path)
        save_json(target_neighbor_diagnostics(prev_aligned_kv, curr_original_kv, curr_aligned_kv, targets, args.target_neighbors_topn), target_neighbors_path)

        outputs = save_aligned_artifacts(curr_aligned_kv, curr_label, output_dir, rotation, targets)
        pair_summary = AlignmentPairSummary(
            source_label=curr_label,
            target_label=prev_label,
            eligible_shared_words=len(ranked_words),
            fit_anchor_words=len(fit_words),
            eval_words=len(eval_words),
            normalize_for_op=args.normalize_for_op,
            objective_space=objective_space,
            fit_eval=fit_diag,
            heldout_eval=heldout_diag,
            files={
                "fit_words_csv": str(fit_csv),
                "eval_words_csv": str(eval_csv),
                "fit_wordwise_cosines_csv": str(fit_cos_csv),
                "heldout_wordwise_cosines_csv": str(eval_cos_csv),
                "fit_top_gains_csv": str(fit_top_gain_csv),
                "fit_worst_after_csv": str(fit_worst_after_csv),
                "heldout_top_gains_csv": str(eval_top_gain_csv),
                "heldout_worst_after_csv": str(eval_worst_after_csv),
                "target_self_cosines_json": str(target_self_path),
                "target_neighbors_json": str(target_neighbors_path),
            },
        )
        save_json(asdict(pair_summary), pair_dir / f"{prev_label}_to_{curr_label}.diagnostics.json")
        save_json(
            {
                "label": curr_label,
                "is_base_label": False,
                "aligned_to_previous_label": prev_label,
                "output_files": outputs,
            },
            output_dir / curr_label / "reports" / f"{curr_label}.alignment_summary.json",
        )
        manifest_rows.append({
            "label": curr_label,
            "is_base_label": False,
            "aligned_to_previous_label": prev_label,
            "eligible_shared_words": len(ranked_words),
            "fit_anchor_words": len(fit_words),
            "eval_words": len(eval_words),
            "fit_mean_cos_before": f"{fit_diag['cosine_before']['mean']:.6f}",
            "fit_mean_cos_after": f"{fit_diag['cosine_after']['mean']:.6f}",
            "heldout_mean_cos_before": f"{heldout_diag['cosine_before']['mean']:.6f}",
            "heldout_mean_cos_after": f"{heldout_diag['cosine_after']['mean']:.6f}",
            "rotation_orthogonality_error_fro": f"{objective_space['rotation_orthogonality_error_fro']:.8f}",
            "cumulative_rotation_to_base": outputs["cumulative_rotation_to_base"],
            "keyed_vectors": outputs["keyed_vectors"],
        })
        prev_aligned_kv = curr_aligned_kv

    manifest_path = output_dir / "alignment_manifest.csv"
    save_csv(manifest_rows, manifest_path, [
        "label", "is_base_label", "aligned_to_previous_label", "eligible_shared_words",
        "fit_anchor_words", "eval_words", "fit_mean_cos_before", "fit_mean_cos_after",
        "heldout_mean_cos_before", "heldout_mean_cos_after", "rotation_orthogonality_error_fro",
        "cumulative_rotation_to_base", "keyed_vectors",
    ])
    phase2_ready = {
        "aligned_run_dir": str(output_dir),
        "ordered_labels": list(labels),
        "embedding_template": "{label}/model/{label}.kv",
        "targets_default": list(targets),
    }
    save_json(phase2_ready, output_dir / "phase2_ready.json")
    run_summary = {
        "alignment_method": "chained_orthogonal_procrustes",
        "base_label": base_label,
        "num_labels_aligned": len(labels),
        "alignment_manifest": str(manifest_path),
        "phase2_ready_json": str(output_dir / "phase2_ready.json"),
        "total_wall_seconds": time.time() - overall_start,
    }
    save_json(run_summary, output_dir / "run_summary.json")
    return {
        "ordered_labels": list(labels),
        "alignment_manifest": str(manifest_path),
        "run_summary": str(output_dir / "run_summary.json"),
        "phase2_ready_json": str(output_dir / "phase2_ready.json"),
    }


def run_displacement(aligned_run_dir: Path, output_dir: Path, labels: Sequence[str], targets: Sequence[str], args: argparse.Namespace) -> Dict[str, object]:
    logging.info("Starting semantic displacement analysis")
    save_json(
        {
            "aligned_run_dir": str(aligned_run_dir),
            "output_dir": str(output_dir),
            "ordered_labels": list(labels),
            "targets": list(targets),
            "pair_min_count": args.pair_min_count,
            "stable_min_count": args.stable_min_count,
            "neighbors_topn": args.neighbors_topn,
            "inspect_top_k_per_pair": args.inspect_top_k_per_pair,
            "inspect_top_k_range": args.inspect_top_k_range,
            "measurement": {
                "primary": "cosine_distance_on_aligned_vectors",
                "adjacent_rate_proxy": "cosine_distance_between_consecutive_labels",
                "whole_range": f"cosine_distance_between_{labels[0]}_and_{labels[-1]}",
            },
        },
        output_dir / "run_config.json",
    )

    overall_start = time.time()
    kv_by_label = {label: load_keyed_vectors(keyed_vectors_path(aligned_run_dir, label)) for label in labels}
    comparison_manifest_rows: List[Dict[str, object]] = []
    pairwise_root = output_dir / "pairwise"
    pairwise_root.mkdir(parents=True, exist_ok=True)

    for left_label, right_label in zip(labels[:-1], labels[1:]):
        logging.info("Computing adjacent displacement: %s_to_%s", left_label, right_label)
        left_kv = kv_by_label[left_label]
        right_kv = kv_by_label[right_label]
        comparison_name = f"{left_label}_to_{right_label}"
        words = shared_words_for_pair(left_kv, right_kv, args.pair_min_count)
        rows_full = build_pairwise_rows(left_kv, right_kv, words, left_label, right_label)
        rows_clean = [r for r in rows_full if bool(r["clean_token"])]
        inspect_tokens = [str(r["token"]) for r in rows_clean[: args.inspect_top_k_per_pair]]
        neighbor_diag = add_top_neighbor_diagnostics(kv_by_label, inspect_tokens, [left_label, right_label], args.neighbors_topn)
        comp_dir = pairwise_root / comparison_name
        comp_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "full_csv": str(comp_dir / f"{comparison_name}.full.csv"),
            "clean_csv": str(comp_dir / f"{comparison_name}.clean.csv"),
            "top100_full_csv": str(comp_dir / f"{comparison_name}.top100_full.csv"),
            "top100_clean_csv": str(comp_dir / f"{comparison_name}.top100_clean.csv"),
            "neighbors_json": str(comp_dir / f"{comparison_name}.neighbors.json"),
            "summary_json": str(comp_dir / f"{comparison_name}.summary.json"),
        }
        summary_obj = comparison_summary(comparison_name, left_label, right_label, rows_full, rows_clean, files)
        write_ranked_pair_outputs(pairwise_root, comparison_name, rows_full, rows_clean, inspect_tokens, neighbor_diag, summary_obj)
        comparison_manifest_rows.append({
            "comparison_name": comparison_name,
            "left_label": left_label,
            "right_label": right_label,
            "n_words_full": len(rows_full),
            "n_words_clean": len(rows_clean),
            "mean_distance_full": f"{summary_obj.distance_stats_full['mean']:.6f}",
            "mean_distance_clean": f"{summary_obj.distance_stats_clean['mean']:.6f}",
            "rho_distance_vs_log_min_count_full": "" if summary_obj.distance_vs_log_min_count_spearman_full["rho"] is None else f"{summary_obj.distance_vs_log_min_count_spearman_full['rho']:.6f}",
            "rho_distance_vs_log_min_count_clean": "" if summary_obj.distance_vs_log_min_count_spearman_clean["rho"] is None else f"{summary_obj.distance_vs_log_min_count_spearman_clean['rho']:.6f}",
            "rho_distance_vs_log_mean_count_full": "" if summary_obj.distance_vs_log_mean_count_spearman_full["rho"] is None else f"{summary_obj.distance_vs_log_mean_count_spearman_full['rho']:.6f}",
            "rho_distance_vs_log_mean_count_clean": "" if summary_obj.distance_vs_log_mean_count_spearman_clean["rho"] is None else f"{summary_obj.distance_vs_log_mean_count_spearman_clean['rho']:.6f}",
            "rho_distance_vs_count_ratio_full": "" if summary_obj.distance_vs_count_ratio_spearman_full["rho"] is None else f"{summary_obj.distance_vs_count_ratio_spearman_full['rho']:.6f}",
            "rho_distance_vs_count_ratio_clean": "" if summary_obj.distance_vs_count_ratio_spearman_clean["rho"] is None else f"{summary_obj.distance_vs_count_ratio_spearman_clean['rho']:.6f}",
            "summary_json": files["summary_json"],
        })

    first_label = labels[0]
    last_label = labels[-1]
    whole_name = f"{first_label}_to_{last_label}"
    logging.info("Computing whole-range displacement: %s", whole_name)
    whole_words = shared_words_for_pair(kv_by_label[first_label], kv_by_label[last_label], args.pair_min_count)
    whole_rows_full = build_pairwise_rows(kv_by_label[first_label], kv_by_label[last_label], whole_words, first_label, last_label)
    whole_rows_clean = [r for r in whole_rows_full if bool(r["clean_token"])]
    whole_inspect_tokens = [str(r["token"]) for r in whole_rows_clean[: args.inspect_top_k_range]]
    whole_neighbor_diag = add_top_neighbor_diagnostics(kv_by_label, whole_inspect_tokens, labels, args.neighbors_topn)
    whole_dir = output_dir / "whole_range"
    whole_dir.mkdir(parents=True, exist_ok=True)
    whole_files = {
        "full_csv": str(whole_dir / f"{whole_name}.full.csv"),
        "clean_csv": str(whole_dir / f"{whole_name}.clean.csv"),
        "top100_full_csv": str(whole_dir / f"{whole_name}.top100_full.csv"),
        "top100_clean_csv": str(whole_dir / f"{whole_name}.top100_clean.csv"),
        "neighbors_json": str(whole_dir / f"{whole_name}.neighbors.json"),
        "summary_json": str(whole_dir / f"{whole_name}.summary.json"),
    }
    whole_summary_obj = comparison_summary(whole_name, first_label, last_label, whole_rows_full, whole_rows_clean, whole_files)
    write_ranked_pair_outputs(output_dir / "whole_range_parent_tmp", whole_name, whole_rows_full, whole_rows_clean, whole_inspect_tokens, whole_neighbor_diag, whole_summary_obj)
    # move from helper-created folder path to desired whole_range layout
    tmp_dir = output_dir / "whole_range_parent_tmp" / whole_name
    for src_name, dest_path in [
        (f"{whole_name}.full.csv", Path(whole_files["full_csv"])),
        (f"{whole_name}.clean.csv", Path(whole_files["clean_csv"])),
        (f"{whole_name}.top100_full.csv", Path(whole_files["top100_full_csv"])),
        (f"{whole_name}.top100_clean.csv", Path(whole_files["top100_clean_csv"])),
        (f"{whole_name}.neighbors.json", Path(whole_files["neighbors_json"])),
        (f"{whole_name}.summary.json", Path(whole_files["summary_json"])),
    ]:
        (tmp_dir / src_name).replace(dest_path)
    tmp_dir.rmdir()
    (output_dir / "whole_range_parent_tmp").rmdir()

    stable_words = shared_words_all_labels(kv_by_label, labels, args.stable_min_count)
    stable_rows_full = build_stable_rows(kv_by_label, labels, stable_words)
    stable_rows_clean = [r for r in stable_rows_full if bool(r["clean_token"])]
    stable_files = write_stable_outputs(output_dir, stable_rows_full, stable_rows_clean, labels, args.inspect_top_k_range, kv_by_label, args.neighbors_topn)

    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    target_rows = target_trajectory_rows(kv_by_label, labels, targets)
    target_fieldnames = ["token", "present_all_labels", "is_probable_name", "is_probable_filler"]
    target_fieldnames += [f"count_{label}" for label in labels]
    target_fieldnames += [f"cos_{left}__to__{right}" for left, right in zip(labels[:-1], labels[1:])]
    target_fieldnames += [f"dist_{left}__to__{right}" for left, right in zip(labels[:-1], labels[1:])]
    target_fieldnames += [
        "range_cosine_similarity", "range_cosine_distance", "adjacent_distance_mean", "adjacent_distance_sum",
        "adjacent_distance_max", "count_ratio_max_to_min", "count_cv_all_labels", "dominant_jump_pair", "one_step_dominance",
    ]
    save_csv(target_rows, targets_dir / "target_trajectories.csv", target_fieldnames)
    save_json(add_top_neighbor_diagnostics(kv_by_label, targets, labels, args.neighbors_topn), targets_dir / "target_neighbors.json")

    interpretation_dir = output_dir / "interpretation"
    interpretation_dir.mkdir(parents=True, exist_ok=True)
    interpretation_payload = {
        "how_to_read_rankings": {
            "primary_measure": "cosine_distance on aligned vectors",
            "highest_distance_words_are_candidates": True,
        },
        "main_confounders": [
            "low_frequency_instability",
            "spelling_or_tokenization_artifacts",
            "proper_names_or_named_entities",
            "genre_composition_shift",
            "polysemy_and_context_mixing",
        ],
        "recommended_words_to_take_seriously": [
            "high distance in clean rankings",
            "adequate counts across relevant labels",
            "reasonable count stability across labels",
            "not dominated by one adjacent jump",
            "coherent nearest-neighbor changes",
            "not obviously a name, filler, or artifact",
        ],
    }
    save_json(interpretation_payload, interpretation_dir / "interpretation_notes.json")

    comparison_manifest_path = output_dir / "comparison_manifest.csv"
    save_csv(comparison_manifest_rows, comparison_manifest_path, [
        "comparison_name", "left_label", "right_label", "n_words_full", "n_words_clean",
        "mean_distance_full", "mean_distance_clean", "rho_distance_vs_log_min_count_full",
        "rho_distance_vs_log_min_count_clean", "rho_distance_vs_log_mean_count_full",
        "rho_distance_vs_log_mean_count_clean", "rho_distance_vs_count_ratio_full",
        "rho_distance_vs_count_ratio_clean", "summary_json",
    ])
    vocabulary_overview = [{"label": label, "vocab_size": len(kv_by_label[label].index_to_key)} for label in labels]
    save_json(vocabulary_overview, output_dir / "vocabulary_overview.json")
    run_summary = {
        "alignment_run_dir": str(aligned_run_dir),
        "output_dir": str(output_dir),
        "ordered_labels": list(labels),
        "num_labels": len(labels),
        "total_wall_seconds": time.time() - overall_start,
        "top_level_files": {
            "run_config_json": str(output_dir / "run_config.json"),
            "comparison_manifest_csv": str(comparison_manifest_path),
            "whole_range_summary_json": str(Path(whole_files["summary_json"])),
            "stable_all_labels_summary_json": stable_files["stable_all_labels_summary_json"],
            "target_trajectories_csv": str(targets_dir / "target_trajectories.csv"),
            "target_neighbors_json": str(targets_dir / "target_neighbors.json"),
            "interpretation_notes_json": str(interpretation_dir / "interpretation_notes.json"),
            "vocabulary_overview_json": str(output_dir / "vocabulary_overview.json"),
        },
    }
    save_json(run_summary, output_dir / "run_summary.json")
    return {
        "comparison_manifest": str(comparison_manifest_path),
        "run_summary": str(output_dir / "run_summary.json"),
    }


def main() -> None:
    args = parse_args()
    input_run_dir = Path(args.input_run_dir)
    alignment_output_dir = Path(args.alignment_output_dir)
    displacement_output_dir = Path(args.displacement_output_dir)
    targets = parse_targets(args.targets)
    labels = discover_bin_labels(input_run_dir, args.bin_labels)

    ensure_output_dir(alignment_output_dir, overwrite=args.overwrite)
    ensure_output_dir(displacement_output_dir, overwrite=args.overwrite)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    logging.info("Input training run dir: %s", input_run_dir)
    logging.info("Ordered labels: %s", labels)
    logging.info("Alignment output dir: %s", alignment_output_dir)
    logging.info("Displacement output dir: %s", displacement_output_dir)

    overall_start = time.time()
    overall_manifest = {
        "input_run_dir": str(input_run_dir),
        "alignment_output_dir": str(alignment_output_dir),
        "displacement_output_dir": str(displacement_output_dir),
        "ordered_labels": list(labels),
        "targets": list(targets),
        "parameters": {
            "alignment": {
                "anchor_min_count": args.anchor_min_count,
                "max_anchor_words": args.max_anchor_words,
                "eval_fraction": args.eval_fraction,
                "min_eval_words": args.min_eval_words,
                "max_eval_words": args.max_eval_words,
                "normalize_for_op": args.normalize_for_op,
                "target_neighbors_topn": args.target_neighbors_topn,
            },
            "displacement": {
                "pair_min_count": args.pair_min_count,
                "stable_min_count": args.stable_min_count,
                "neighbors_topn": args.neighbors_topn,
                "inspect_top_k_per_pair": args.inspect_top_k_per_pair,
                "inspect_top_k_range": args.inspect_top_k_range,
            },
            "overwrite": args.overwrite,
        },
    }
    save_json(overall_manifest, alignment_output_dir / "align_and_displace_manifest.json")

    alignment_result = run_alignment(input_run_dir, alignment_output_dir, labels, targets, args)
    displacement_result = run_displacement(alignment_output_dir, displacement_output_dir, labels, targets, args)

    overall_summary = {
        "input_run_dir": str(input_run_dir),
        "alignment_output_dir": str(alignment_output_dir),
        "displacement_output_dir": str(displacement_output_dir),
        "ordered_labels": list(labels),
        "alignment_summary": alignment_result,
        "displacement_summary": displacement_result,
        "total_wall_seconds": time.time() - overall_start,
    }
    save_json(overall_summary, alignment_output_dir / "align_and_displace_summary.json")
    logging.info("Alignment and displacement completed successfully.")


if __name__ == "__main__":
    main()
