# What-is-love
Language Technology project about the shift in the semantic field of love in song lyrics

---

- **uv**: a fast Python package manager and virtual-environment tool.
- **virtual environment**: an isolated Python installation for this project only.
- **`pyproject.toml`**: the project metadata and dependency declaration.
- **`uv.lock`**: the resolved dependency versions for the project.
- **`requirements-hpc.txt`**: the frozen package list exported from the working HPC environment and used here as the practical installation target on the cluster.

The repository is published as an **HPC-ready skeleton**:
- code, SLURM scripts, lightweight results, and folder structure are tracked on GitHub;
- raw parquet data, prepared token files, and heavy model binaries are **not** tracked and must be restored from the shared Google Drive before running the full pipeline.

---

## 1. Clone the repository

Choose a folder in your HPC home directory and clone the repo there.

```bash
git clone https://github.com/TakPit/What-is-love.git
cd What-is-love
````

Optional but recommended if you plan to commit changes:

```bash
git config user.name "Your Name"
git config user.email "your_email@example.com"
```

---

## 2. Load Python 3.11 on the HPC

This project expects Python 3.11.

If your cluster uses environment modules, load Python first. The exact module name may differ on your system.

```bash
module load python/3.11
python --version
```

If your cluster does not use modules, just make sure `python` or `python3.11` points to Python 3.11.

---

## 3. Install `uv` on the HPC

Install `uv` in your user space.

```bash
python -m pip install --user --upgrade pip uv
```

Make sure your local user bin directory is on `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If that works, add the same line to your shell startup file so it is available in future sessions:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. Create the project environment

This repository keeps `pyproject.toml` and `uv.lock` for reproducibility, but on this HPC the operational installation path is:

1. create a named virtual environment with `uv`;
2. install the frozen package set from `requirements-hpc.txt`.

This is the environment setup that should match the published project state.

### 4.1 Create the environment

```bash
uv venv --python 3.11 nlp_proj_env
source nlp_proj_env/bin/activate
```

### 4.2 Install the packages

```bash
uv pip install -r requirements-hpc.txt
```

### 4.3 Verify the environment

```bash
python -c "import numpy, pandas, scipy, gensim, matplotlib; print('environment ok')"
```

### Important notes

* Always activate the environment with:

  ```bash
  source nlp_proj_env/bin/activate
  ```

* Do **not** run `uv add ...` in this repository unless you are intentionally changing dependencies.

* Do **not** create or use a separate `.venv` for routine work in this project.

* The intended environment name is `nlp_proj_env`.

---

## 5. Restore the external data from Google Drive

Three heavy data/model components are distributed outside GitHub and must be placed back into the repository skeleton.

### 5.1 Raw parquet dataset

Download all parquet shards from the shared Drive and place them here:

```text
data/genius-lyrics-cleaned/
```

After this step, that folder should contain:

```text
train-00000-of-00010.parquet
...
train-00009-of-00010.parquet
```

### 5.2 Prepared tokenized corpora

Download the contents of:

```text
results/phase1/corpus_preparation_and_exploration/1960_1970_1980_1990_2000_2010_2020/sentences_by_bin/
```

and place the files back into the same location in your local clone.

That folder should then contain:

```text
1960_1969.txt.gz
1970_1979.txt.gz
1980_1989.txt.gz
1990_1999.txt.gz
2000_2009.txt.gz
2010_2019.txt.gz
```

### 5.3 Training model contents

Download the model contents for each label under:

```text
results/phase1/training/sgns/1960_1970_1980_1990_2000_2010_2020/
```

and restore the files inside each label’s `model/` subfolder.

Concretely, for each of:

```text
1960_1969
1970_1979
1980_1989
1990_1999
2000_2009
2010_2019
```

put the provided model files into:

```text
results/phase1/training/sgns/1960_1970_1980_1990_2000_2010_2020/<LABEL>/model/
```

Do **not** delete the lightweight report files already present in the repository.

---

## 6. One-time SLURM editing checklist

Before launching jobs, inspect the SLURM scripts under:

```text
slurm/phase1/
slurm/phase2/
```

### 6.1 Replace user-specific paths

Some scripts may contain user-specific absolute paths from the original setup.

Search for anything like:

* `3192105`
* `/home/3192105/`
* old repo folder names

Useful search commands:

```bash
grep -R "3192105" slurm src
grep -R "/home/" slurm src
grep -R "what_is_love_publish" slurm src
```

Update those paths so they point to **your own clone location**.

### 6.2 Check cluster-specific `#SBATCH` settings

Each teammate should verify that the following match their HPC account and cluster rules:

* account / project
* partition / queue
* time limit
* CPUs
* memory
* GPU requests, if any
* output and error log paths

### 6.3 Check run/config variables

Do not change these unless you are intentionally running an ablation or a different experiment:

* `BIN_EDGES`
* `RUN_ID`
* config file path for Phase 2
* thresholds such as `min_count`
* pairwise or neighbor settings in Phase 2

For the baseline published run, these should remain aligned with:

```text
1960_1970_1980_1990_2000_2010_2020
```

---

## 7. Canonical pipeline and where to start

There are three valid starting points depending on what you already restored from Drive.

---

### Option A. Full rerun from raw parquet files

Use this if you want to rerun everything from the raw corpus.

Run in this order:

```bash
sbatch slurm/phase1/corpus_preparation_and_exploration.sh
sbatch slurm/phase1/train_sgns_decades.sh
sbatch slurm/phase1/align_and_displace.sh
sbatch slurm/phase2/semantic_axes_preparation.sh
sbatch slurm/phase2/run_semantic_axes.sh
```

---

### Option B. Start from prepared tokenized corpora

Use this if you restored `sentences_by_bin/` and want to skip the corpus-preparation stage.

Run:

```bash
sbatch slurm/phase1/train_sgns_decades.sh
sbatch slurm/phase1/align_and_displace.sh
sbatch slurm/phase2/semantic_axes_preparation.sh
sbatch slurm/phase2/run_semantic_axes.sh
```

---

### Option C. Start from restored training models

Use this if you restored the training model contents and want to skip both corpus preparation and SGNS training.

Run:

```bash
sbatch slurm/phase1/align_and_displace.sh
sbatch slurm/phase2/semantic_axes_preparation.sh
sbatch slurm/phase2/run_semantic_axes.sh
```

---

## 8. What each stage does

### Phase 1

#### `slurm/phase1/corpus_preparation_and_exploration.sh`

Runs:

```text
src/phase1/corpus_preparation_and_exploration.py
```

Purpose:

* reads parquet lyrics
* cleans/tokenizes
* bins the corpus into the published time slices
* writes lightweight corpus-preparation summaries

Main output root:

```text
results/phase1/corpus_preparation_and_exploration/1960_1970_1980_1990_2000_2010_2020/
```

#### `slurm/phase1/train_sgns_decades.sh`

Runs:

```text
src/phase1/train_sgns_decades.py
```

Purpose:

* trains one SGNS (skip-gram with negative sampling) model per time bin

Main output root:

```text
results/phase1/training/sgns/1960_1970_1980_1990_2000_2010_2020/
```

#### `slurm/phase1/align_and_displace.sh`

Runs:

```text
src/phase1/align_and_displace.py
```

Purpose:

* aligns the time-slice embedding spaces
* computes semantic displacement

Main output roots:

```text
results/phase1/alignment/sgns/1960_1970_1980_1990_2000_2010_2020/
results/phase1/displacement/sgns/1960_1970_1980_1990_2000_2010_2020/
```

### Phase 2

#### `slurm/phase2/semantic_axes_preparation.sh`

Runs:

```text
src/phase2/semantic_axes_preparation.py
```

Purpose:

* checks whether targets and anchor words are covered in the aligned embeddings
* prepares the item summary used by the axis run

Main output root:

```text
results/phase2/semantic_axes_preparation/
```

#### `slurm/phase2/run_semantic_axes.sh`

Runs:

```text
src/phase2/run_semantic_axes.py
```

Purpose:

* builds the semantic axes by label
* scores targets against those axes
* writes summaries and plots

Main output root:

```text
results/phase2/semantic_axes/
```

---

## 9. Minimal sanity checks after each stage

### After corpus preparation

Check that this folder exists and contains the expected summaries:

```text
results/phase1/corpus_preparation_and_exploration/1960_1970_1980_1990_2000_2010_2020/
```

Important files:

* `bin_summary.csv`
* `preparation_summary.json`
* `sanity_summary.json`
* `target_word_counts_by_bin.csv`

### After training

Check:

```text
results/phase1/training/sgns/1960_1970_1980_1990_2000_2010_2020/
```

Important files:

* `training_manifest.csv`
* `run_summary.json`
* per-label `reports/*.summary.json`
* per-label `reports/*.vocab.csv`

### After alignment/displacement

Check:

```text
results/phase1/alignment/sgns/1960_1970_1980_1990_2000_2010_2020/
results/phase1/displacement/sgns/1960_1970_1980_1990_2000_2010_2020/
```

Important files:

* `phase2_ready.json`
* `alignment_manifest.csv`
* `comparison_manifest.csv`
* `run_summary.json`

### After semantic-axis preparation

Check:

```text
results/phase2/semantic_axes_preparation/config1/
```

Important files:

* `config1.item_summary.csv`
* `config1.coverage_by_label.csv`

### After semantic-axis run

Check:

```text
results/phase2/semantic_axes/config1/
```

Important files:

* `config1.run.json`
* `config1.axes_by_label.json`
* `config1.target_axis_scores.csv`
* `config1.field_summary.csv`
* the three PNG plots

---

## 10. Reading job logs

Logs go to:

```text
logs/out/
logs/err/
```

Typical pattern:

```text
logs/out/<job_name>_<jobid>.out
logs/err/<job_name>_<jobid>.err
```

Inspect them with:

```bash
less logs/out/<filename>.out
less logs/err/<filename>.err
```

---

## 11. Contribution workflow for teammates

If you only want to run experiments locally, you do not need to push anything.

If you want to contribute code or documentation changes back to GitHub:

### 11.1 Create a branch

```bash
git checkout -b feature/my-change
```

### 11.2 Make your changes

Typical tracked changes:

* `src/`
* `slurm/`
* `README.md`
* lightweight JSON/CSV outputs, if the group explicitly agrees to update them

Do **not** commit:

* raw parquet files
* `sentences_by_bin/*.txt.gz`
* heavy model files
* logs
* local virtual environments

### 11.3 Review what changed

```bash
git status
find . -type f -size +50M
```

### 11.4 Commit

```bash
git add .
git commit -m "Describe the change clearly"
```

### 11.5 Push with your own GitHub credentials

This repository currently uses **HTTPS**, not SSH.

So when you run:

```bash
git push origin feature/my-change
```

and Git asks for credentials:

* **Username** = your GitHub username
* **Password** = your own GitHub Personal Access Token

Do **not** use:

* somebody else’s credentials
* an SSH key pasted into the password prompt

### 11.6 Open a pull request

Push your branch and then open a pull request into `main`.

---

## 12. Common pitfalls

### Wrong Python version

This project expects Python 3.11. Check with:

```bash
python --version
```

### Accidentally creating `.venv`

Do not use a separate `.venv` for routine work here. Use the published environment name:

```text
nlp_proj_env
```

### Forgetting to activate the environment

Always run:

```bash
source nlp_proj_env/bin/activate
```

before interactive Python work.

### Wrong paths in SLURM files

Each teammate should search for and replace hard-coded user paths before the first run.

### Missing Drive content

If a stage fails because a file is missing, first check whether the corresponding Drive-delivered content was restored to the correct folder.

### Old label format

The current published bins are:

```text
1960_1969
1970_1979
1980_1989
1990_1999
2000_2009
2010_2019
```

Do not switch back to old-style labels such as `1970s` unless you intentionally update the whole pipeline.

---

## 13. Baseline run IDs and config names

Published baseline run identifiers:

### Phase 1

```text
1960_1970_1980_1990_2000_2010_2020
```

### Phase 2

```text
config1
```

Unless you are intentionally running an ablation, keep these unchanged.

---

## 14. Final baseline launch order

For the baseline project, the canonical order is:

```bash
sbatch slurm/phase1/corpus_preparation_and_exploration.sh
sbatch slurm/phase1/train_sgns_decades.sh
sbatch slurm/phase1/align_and_displace.sh
sbatch slurm/phase2/semantic_axes_preparation.sh
sbatch slurm/phase2/run_semantic_axes.sh
```

If you restored prepared tokens, skip the first step.
If you restored training model contents, skip the first two steps.

---

## 15. Repository structure expected by the pipeline

```text
data/genius-lyrics-cleaned/
results/phase1/corpus_preparation_and_exploration/1960_1970_1980_1990_2000_2010_2020/
results/phase1/training/sgns/1960_1970_1980_1990_2000_2010_2020/
results/phase1/alignment/sgns/1960_1970_1980_1990_2000_2010_2020/
results/phase1/displacement/sgns/1960_1970_1980_1990_2000_2010_2020/
results/phase2/semantic_axes_preparation/
results/phase2/semantic_axes/
slurm/phase1/
slurm/phase2/
src/phase1/
src/phase2/
```

Keep this structure unchanged unless the group explicitly agrees on a repo-wide refactor.

```
```
