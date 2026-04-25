#!/bin/bash

#SBATCH --job-name="corpus_prep_explore"
#SBATCH --account=3192105
#SBATCH --partition=stud
#SBATCH --qos=stud

#SBATCH --chdir=/home/3192105/nlp_project

#SBATCH --output=logs/out/%x_%j.out
#SBATCH --error=logs/err/%x_%j.err

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

cd /home/3192105/nlp_project

# ------------------------------------------------------------
# Edit this line to change the time bins.
# Interpreted as half-open bins:
#   [1960,1970), [1970,1980), ... [2010,2020)
# which correspond to 1960_1969, 1970_1979, ..., 2010_2019
# ------------------------------------------------------------
BIN_EDGES="1960,1970,1980,1990,2000,2010,2020"
RUN_ID="$(echo "${BIN_EDGES}" | tr ',' '_')"
OUT_DIR="results/phase1/corpus_preparation_and_exploration/${RUN_ID}"

mkdir -p "${OUT_DIR}"

source nlp_proj_env/bin/activate

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
echo "BIN_EDGES=${BIN_EDGES}"
echo "OUT_DIR=${OUT_DIR}"

python src/phase1/corpus_preparation_and_exploration.py \
  --input-dir data/genius-lyrics-cleaned \
  --output-dir "${OUT_DIR}" \
  --bin-edges "${BIN_EDGES}" \
  --targets "love,heart,baby,kiss,touch,forever,mine,desire,hurt,broken" \
  --thresholds "5,10,20,50" \
  --top-n 100 \
  --dedup-adjacent-lines

