#!/bin/bash

#SBATCH --job-name="train_sgns_bins"
#SBATCH --account=3192105
#SBATCH --partition=stud
#SBATCH --qos=stud

#SBATCH --chdir=/home/3192105/nlp_project

#SBATCH --output=logs/out/%x_%j.out
#SBATCH --error=logs/err/%x_%j.err

#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

set -euo pipefail

cd /home/3192105/nlp_project

# ------------------------------------------------------------
# Keep this consistent with corpus_preparation_and_exploration.sh
# so the training run name and the preparation run name match.
# ------------------------------------------------------------
BIN_EDGES="1960,1970,1980,1990,2000,2010,2020"
RUN_ID="$(echo "${BIN_EDGES}" | tr ',' '_')"

# Optional: leave empty to train all prepared bins discovered in the folder.
# Example subset: BIN_LABELS="1970_1979,1980_1989,1990_1999"
BIN_LABELS=""

PREP_DIR="results/phase1/corpus_preparation_and_exploration/${RUN_ID}"
INPUT_DIR="${PREP_DIR}/sentences_by_bin"
OUTPUT_ROOT="results/phase1/training/sgns"

mkdir -p "${OUTPUT_ROOT}"

source nlp_proj_env/bin/activate

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
echo "BIN_EDGES=${BIN_EDGES}"
echo "RUN_ID=${RUN_ID}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

CMD=(
  python src/phase1/train_sgns_decades.py
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_ROOT}"
  --run-name "${RUN_ID}"
  --vector-size 300
  --window 4
  --min-count 20
  --epochs 5
  --negative 5
  --sample 1e-4
  --workers 4
)

if [ -n "${BIN_LABELS}" ]; then
  CMD+=(--bin-labels "${BIN_LABELS}")
fi

"${CMD[@]}"
