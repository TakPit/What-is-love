#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


BIN_FILE_RE = re.compile(r"^(?P<label>\d{4}_\d{4})\.txt\.gz$")


class SentenceCorpus:
    """
    Re-iterable corpus reader for a single prepared bin file.

    Each line of the input .txt.gz file is already a tokenized lyric line,
    with whitespace-separated tokens. This iterator yields one list[str]
    per line, which is exactly the input format gensim's Word2Vec expects.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[List[str]]:
        with gzip.open(self.path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield line.split()


class EpochLogger(CallbackAny2Vec):
    """
    Small gensim callback to make training logs easier to read.

    Optionally records training loss if compute_loss=True is enabled.
    """

    def __init__(self, bin_label: str, compute_loss: bool = False):
        self.bin_label = bin_label
        self.compute_loss = compute_loss
        self.epoch = 0
        self.prev_loss = 0.0
        self.epoch_losses: List[float] = []

    def on_epoch_begin(self, model):
        logging.info("[%s] Starting epoch %d", self.bin_label, self.epoch + 1)

    def on_epoch_end(self, model):
        if self.compute_loss:
            latest_loss = model.get_latest_training_loss()
            epoch_loss = latest_loss - self.prev_loss
            self.prev_loss = latest_loss
            self.epoch_losses.append(epoch_loss)
            logging.info(
                "[%s] Finished epoch %d | incremental_loss=%.4f",
                self.bin_label,
                self.epoch + 1,
                epoch_loss,
            )
        else:
            logging.info("[%s] Finished epoch %d", self.bin_label, self.epoch + 1)
        self.epoch += 1


@dataclass
class RawCorpusStats:
    """Simple corpus statistics computed before vocabulary pruning."""

    line_count: int
    token_count: int
    avg_line_length: float
    min_line_length: int
    max_line_length: int


@dataclass
class BinTrainingSummary:
    """Summary saved per bin after training."""

    bin_label: str
    source_file: str
    raw_stats: Dict[str, float]
    vocab_size_after_min_count: int
    corpus_count_seen_by_gensim: int
    corpus_total_words_seen_by_gensim: int
    training_seconds: float
    model_files: Dict[str, str]
    target_neighbors_file: Optional[str]
    epoch_losses: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SGNS / word2vec models from prepared lyric bins."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing files like 1960_1969.txt.gz, 1970_1979.txt.gz, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase1/training/sgns",
        help="Root directory under which run_name/ will be created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Unique run identifier. If omitted, the script uses the parent folder name "
            "of --input-dir, which keeps it consistent with the corpus-preparation run."
        ),
    )
    parser.add_argument(
        "--bin-labels",
        type=str,
        default="",
        help=(
            "Optional comma-separated subset of prepared bin labels to train, e.g. "
            "1960_1969,1970_1979. Leave empty to auto-discover and train all bins."
        ),
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=300,
        help="Embedding dimensionality.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="Context window size on each side.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum token frequency for vocabulary inclusion.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of SGNS training epochs.",
    )
    parser.add_argument(
        "--negative",
        type=int,
        default=5,
        help="Number of negative samples for SGNS.",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1e-4,
        help="Subsampling threshold for very frequent words.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of CPU worker threads for gensim.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization.",
    )
    parser.add_argument(
        "--dynamic-window",
        action="store_true",
        help=(
            "Use word2vec-style dynamic window shrinking. Leave off for a more literal "
            "fixed-window baseline."
        ),
    )
    parser.add_argument(
        "--compute-loss",
        action="store_true",
        help="Track training loss per epoch. Slightly slower, but useful for diagnostics.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="love,heart,baby,kiss,touch,forever,mine,desire,hurt,broken",
        help="Comma-separated target words for quick nearest-neighbor inspection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run directory.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_targets(value: str) -> List[str]:
    targets = [t.strip().lower() for t in value.split(",") if t.strip()]
    return sorted(set(targets))


def infer_run_name(input_dir: Path, explicit_run_name: str) -> str:
    if explicit_run_name.strip():
        return explicit_run_name.strip()
    parent_name = input_dir.parent.name.strip()
    if not parent_name:
        raise ValueError("Could not infer run name from --input-dir parent. Pass --run-name explicitly.")
    return parent_name


def bin_sort_key(label: str) -> tuple[int, str]:
    match = re.match(r"^(\d{4})_(\d{4})$", label)
    if match:
        return (int(match.group(1)), label)
    return (10**9, label)


def discover_bin_files(input_dir: Path) -> Dict[str, Path]:
    bin_files: Dict[str, Path] = {}
    for path in sorted(input_dir.glob("*.txt.gz")):
        match = BIN_FILE_RE.match(path.name)
        if not match:
            continue
        label = match.group("label")
        bin_files[label] = path
    if not bin_files:
        raise FileNotFoundError(f"No bin files matching YYYY_YYYY.txt.gz found in: {input_dir}")
    return bin_files


def select_bin_labels(discovered_bin_files: Dict[str, Path], requested_bin_labels: str) -> List[str]:
    discovered = sorted(discovered_bin_files.keys(), key=bin_sort_key)
    if not requested_bin_labels.strip():
        return discovered

    requested = parse_csv_list(requested_bin_labels)
    missing = [label for label in requested if label not in discovered_bin_files]
    if missing:
        raise FileNotFoundError(f"Requested bin labels are missing from input-dir: {missing}")
    return sorted(requested, key=bin_sort_key)


def compute_raw_corpus_stats(path: Path) -> RawCorpusStats:
    """
    Compute lightweight raw stats before training.

    These are useful for logging and for comparing runs later, but they do not
    duplicate the full corpus-preparation sanity report. We keep this pass light.
    """
    line_count = 0
    token_count = 0
    min_line_length: Optional[int] = None
    max_line_length = 0

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            n = len(tokens)
            line_count += 1
            token_count += n
            if min_line_length is None or n < min_line_length:
                min_line_length = n
            if n > max_line_length:
                max_line_length = n

    if line_count == 0:
        raise ValueError(f"Prepared bin file is empty after reading tokenized lines: {path}")

    avg_line_length = token_count / line_count
    return RawCorpusStats(
        line_count=line_count,
        token_count=token_count,
        avg_line_length=avg_line_length,
        min_line_length=min_line_length or 0,
        max_line_length=max_line_length,
    )


def save_vocab_counts_csv(model: Word2Vec, output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "token", "count"])
        writer.writeheader()
        for rank, token in enumerate(model.wv.index_to_key, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "token": token,
                    "count": model.wv.get_vecattr(token, "count"),
                }
            )


def save_target_neighbors(model: Word2Vec, targets: List[str], output_path: Path, topn: int = 20) -> None:
    payload: Dict[str, List[List[float | str]]] = {}
    for target in targets:
        if target not in model.wv:
            payload[target] = []
            continue
        payload[target] = [[word, float(score)] for word, score in model.wv.most_similar(target, topn=topn)]

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def ensure_run_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {path}\n"
            "Use a new --run-name or pass --overwrite if you really want to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    run_name = infer_run_name(input_dir, args.run_name)
    run_dir = output_root / run_name
    targets = parse_targets(args.targets)

    ensure_run_directory(run_dir, overwrite=args.overwrite)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    logging.info("Starting SGNS training run: %s", run_name)
    logging.info("Input dir:  %s", input_dir)
    logging.info("Output dir: %s", run_dir)

    bin_files = discover_bin_files(input_dir)
    selected_bin_labels = select_bin_labels(bin_files, args.bin_labels)

    config = {
        "run_name": run_name,
        "prep_run_id": input_dir.parent.name,
        "input_dir": str(input_dir),
        "output_dir": str(run_dir),
        "selected_bins": selected_bin_labels,
        "bin_files": {label: str(bin_files[label]) for label in selected_bin_labels},
        "targets": targets,
        "sgns_hyperparameters": {
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "negative": args.negative,
            "sample": args.sample,
            "workers": args.workers,
            "seed": args.seed,
            "sg": 1,
            "dynamic_window": args.dynamic_window,
            "compute_loss": args.compute_loss,
        },
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    training_manifest_rows: List[Dict[str, object]] = []
    overall_start = time.time()

    for bin_label in selected_bin_labels:
        source_path = bin_files[bin_label]
        bin_dir = run_dir / bin_label
        model_dir = bin_dir / "model"
        report_dir = bin_dir / "reports"
        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        logging.info("=" * 80)
        logging.info("Training bin: %s", bin_label)
        logging.info("Source file: %s", source_path)

        raw_stats = compute_raw_corpus_stats(source_path)
        logging.info(
            "[%s] Raw corpus | lines=%d | tokens=%d | avg_line_len=%.2f",
            bin_label,
            raw_stats.line_count,
            raw_stats.token_count,
            raw_stats.avg_line_length,
        )

        corpus_for_vocab = SentenceCorpus(source_path)
        epoch_logger = EpochLogger(bin_label=bin_label, compute_loss=args.compute_loss)
        model = Word2Vec(
            sg=1,
            hs=0,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            negative=args.negative,
            sample=args.sample,
            seed=args.seed,
            compute_loss=args.compute_loss,
            shrink_windows=args.dynamic_window,
            sorted_vocab=1,
        )

        build_start = time.time()
        model.build_vocab(corpus_for_vocab)
        build_seconds = time.time() - build_start
        logging.info(
            "[%s] Vocabulary built | retained_vocab=%d | corpus_count=%d | corpus_total_words=%d | build_sec=%.2f",
            bin_label,
            len(model.wv),
            model.corpus_count,
            model.corpus_total_words,
            build_seconds,
        )

        corpus_for_training = SentenceCorpus(source_path)
        train_start = time.time()
        model.train(
            corpus_iterable=corpus_for_training,
            total_examples=model.corpus_count,
            epochs=args.epochs,
            callbacks=[epoch_logger],
        )
        train_seconds = time.time() - train_start
        logging.info("[%s] Training completed | train_sec=%.2f", bin_label, train_seconds)

        model_path = model_dir / f"{bin_label}.model"
        keyed_vectors_path = model_dir / f"{bin_label}.kv"
        vectors_txt_path = model_dir / f"{bin_label}.vectors.txt"
        vectors_bin_path = model_dir / f"{bin_label}.vectors.bin"
        vocab_csv_path = report_dir / f"{bin_label}.vocab.csv"
        neighbors_json_path = report_dir / f"{bin_label}.target_neighbors.json"
        summary_json_path = report_dir / f"{bin_label}.summary.json"

        model.save(str(model_path))
        model.wv.save(str(keyed_vectors_path))
        model.wv.save_word2vec_format(str(vectors_txt_path), binary=False)
        model.wv.save_word2vec_format(str(vectors_bin_path), binary=True)
        save_vocab_counts_csv(model, vocab_csv_path)
        save_target_neighbors(model, targets, neighbors_json_path)

        bin_summary = BinTrainingSummary(
            bin_label=bin_label,
            source_file=str(source_path),
            raw_stats=asdict(raw_stats),
            vocab_size_after_min_count=len(model.wv),
            corpus_count_seen_by_gensim=model.corpus_count,
            corpus_total_words_seen_by_gensim=model.corpus_total_words,
            training_seconds=train_seconds,
            model_files={
                "model": str(model_path),
                "keyed_vectors": str(keyed_vectors_path),
                "vectors_txt": str(vectors_txt_path),
                "vectors_bin": str(vectors_bin_path),
                "vocab_csv": str(vocab_csv_path),
            },
            target_neighbors_file=str(neighbors_json_path),
            epoch_losses=epoch_logger.epoch_losses,
        )
        with open(summary_json_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(bin_summary), fh, indent=2)

        training_manifest_rows.append(
            {
                "bin_label": bin_label,
                "source_file": str(source_path),
                "raw_lines": raw_stats.line_count,
                "raw_tokens": raw_stats.token_count,
                "avg_line_length": f"{raw_stats.avg_line_length:.4f}",
                "retained_vocab": len(model.wv),
                "corpus_count": model.corpus_count,
                "corpus_total_words": model.corpus_total_words,
                "build_seconds": f"{build_seconds:.4f}",
                "train_seconds": f"{train_seconds:.4f}",
                "summary_json": str(summary_json_path),
            }
        )

    manifest_path = run_dir / "training_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "bin_label",
                "source_file",
                "raw_lines",
                "raw_tokens",
                "avg_line_length",
                "retained_vocab",
                "corpus_count",
                "corpus_total_words",
                "build_seconds",
                "train_seconds",
                "summary_json",
            ],
        )
        writer.writeheader()
        for row in training_manifest_rows:
            writer.writerow(row)

    overall_summary = {
        "run_name": run_name,
        "prep_run_id": input_dir.parent.name,
        "selected_bins": selected_bin_labels,
        "num_bins_trained": len(selected_bin_labels),
        "total_wall_seconds": time.time() - overall_start,
        "training_manifest": str(manifest_path),
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(overall_summary, fh, indent=2)

    logging.info("=" * 80)
    logging.info("Run completed successfully.")
    logging.info("Manifest: %s", manifest_path)
    logging.info("Run summary: %s", run_dir / "run_summary.json")


if __name__ == "__main__":
    main()
