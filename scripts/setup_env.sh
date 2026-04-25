#!/bin/bash
set -euo pipefail

ENV_NAME="nlp_proj_env"

python3.11 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements-hpc.txt

python -c "import numpy, pandas, scipy, gensim, matplotlib; print('environment ok')"
echo "[ok] Environment ready: $ENV_NAME"
