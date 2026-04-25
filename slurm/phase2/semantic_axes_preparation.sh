#!/bin/bash

#SBATCH --job-name="semantic_axes_preparation"
#SBATCH --account=3192105
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --chdir=/home/3192105/nlp_project
#SBATCH --output=/home/3192105/nlp_project/logs/out/semantic_axes_preparation_%j.out
#SBATCH --error=/home/3192105/nlp_project/logs/err/semantic_axes_preparation_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="/home/3192105/nlp_project"
cd "$REPO_DIR"

source nlp_proj_env/bin/activate

RUN_ID="1960_1970_1980_1990_2000_2010_2020"
CONFIG="results/phase2/semantic_axes_preparation/semantic_axes_config1.json"
PHASE1_READY_JSON="results/phase1/alignment/sgns/${RUN_ID}/phase2_ready.json"
OUTPUT_ROOT="results/phase2/semantic_axes_preparation"
OVERWRITE=1

CMD=(
  python src/phase2/semantic_axes_preparation.py
  --config "$CONFIG"
  --phase1-ready-json "$PHASE1_READY_JSON"
  --output-root "$OUTPUT_ROOT"
)

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

"${CMD[@]}"
