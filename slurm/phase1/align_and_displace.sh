#!/bin/bash

#SBATCH --job-name="align_and_displace"
#SBATCH --account=3192105
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --chdir=/home/3192105/nlp_project
#SBATCH --output=/home/3192105/nlp_project/logs/out/align_and_displace_%j.out
#SBATCH --error=/home/3192105/nlp_project/logs/err/align_and_displace_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00

set -euo pipefail

cd /home/3192105/nlp_project
source nlp_proj_env/bin/activate

# Keep this identical to the prep/training run identity.
BIN_EDGES="1960,1970,1980,1990,2000,2010,2020"
RUN_ID="${BIN_EDGES//,/_}"

# Optional manual override for label order.
# Leave empty to auto-discover labels from the training run metadata/folders.
BIN_LABELS=""

TARGETS="love,heart,baby,kiss,touch,forever,mine,desire,hurt,broken"

# Alignment settings.
ANCHOR_MIN_COUNT=100
MAX_ANCHOR_WORDS=20000
EVAL_FRACTION=0.10
MIN_EVAL_WORDS=500
MAX_EVAL_WORDS=2000
NORMALIZE_FOR_OP=1
TARGET_NEIGHBORS_TOPN=20

# Displacement settings.
PAIR_MIN_COUNT=100
STABLE_MIN_COUNT=100
NEIGHBORS_TOPN=20
INSPECT_TOP_K_PER_PAIR=50
INSPECT_TOP_K_RANGE=100

# Safety.
OVERWRITE=0

INPUT_RUN_DIR="results/phase1/training/sgns/${RUN_ID}"
ALIGNMENT_OUTPUT_DIR="results/phase1/alignment/sgns/${RUN_ID}"
DISPLACEMENT_OUTPUT_DIR="results/phase1/displacement/sgns/${RUN_ID}"

mkdir -p "results/phase1/alignment/sgns"
mkdir -p "results/phase1/displacement/sgns"

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${ALIGNMENT_OUTPUT_DIR}" "${DISPLACEMENT_OUTPUT_DIR}"
fi

CMD=(
  python src/phase1/align_and_displace.py
  --input-run-dir "${INPUT_RUN_DIR}"
  --alignment-output-dir "${ALIGNMENT_OUTPUT_DIR}"
  --displacement-output-dir "${DISPLACEMENT_OUTPUT_DIR}"
  --targets "${TARGETS}"
  --anchor-min-count "${ANCHOR_MIN_COUNT}"
  --max-anchor-words "${MAX_ANCHOR_WORDS}"
  --eval-fraction "${EVAL_FRACTION}"
  --min-eval-words "${MIN_EVAL_WORDS}"
  --max-eval-words "${MAX_EVAL_WORDS}"
  --target-neighbors-topn "${TARGET_NEIGHBORS_TOPN}"
  --pair-min-count "${PAIR_MIN_COUNT}"
  --stable-min-count "${STABLE_MIN_COUNT}"
  --neighbors-topn "${NEIGHBORS_TOPN}"
  --inspect-top-k-per-pair "${INSPECT_TOP_K_PER_PAIR}"
  --inspect-top-k-range "${INSPECT_TOP_K_RANGE}"
)

if [[ -n "${BIN_LABELS}" ]]; then
  CMD+=(--bin-labels "${BIN_LABELS}")
fi

if [[ "${NORMALIZE_FOR_OP}" == "1" ]]; then
  CMD+=(--normalize-for-op)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

"${CMD[@]}"
