# Accountability – Media Criticism Pipeline

## Overview

This repository implements an LLM-based pipeline that analyzes Swiss media
coverage of public administration to detect criticism of federal
departments, administrative units, independent agencies and federal
councillors, and to characterize that criticism (who voices it, what it
targets, whether the administration responds).

The pipeline runs as a chain of SLURM batch jobs on the UNIL cluster, each
step consuming the merged output of the previous one.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| download | `scripts/download.py` | Fetches raw articles from Swissdox |
| 2 | `scripts/step2_clean_targets.py` | Dedupes articles, matches every target (department, unit, agency, councillor) mentioned, explodes to one row per (article, target) |
| 3 | `scripts/step3_criticism_detection.py` | LLM run: is this (article, target) row a criticism? |
| 4a | `scripts/step4a_source_attribution.py` | LLM run: is the criticism attributed to a named actor, or is it journalist framing? |
| 4b | `scripts/step4b_source_category.py` | LLM run: classifies the sourced actor into one of 6 categories |
| 5 | `scripts/step5_content_type.py` | LLM run: is the criticism about a policy choice or about the target as an entity? |
| 6 | `scripts/step6_admin_response.py` | LLM run: does the target respond to the criticism in the article? |

Each `sbatch_*.sh` / `sbatch_*_array.sh` pair submits one step (single job or
array job); `sbatch_merge_step*.sh` merges the array's shards. `launch_pipeline.sh`
chains all of the above automatically via `--dependency=afterok`.

## Repository Structure

```
ACCOUNTABILITY_REPO/
├── requirements.txt
├── launch_pipeline.sh          # chains all SLURM steps end to end
├── sbatch_download.sh
├── sbatch_step2.sh
├── sbatch_step{3,4a,4b,5,6}_array.sh
├── sbatch_merge_step{3,4a,4b,5,6}.sh
├── scripts/                    # one entry-point script per pipeline step
├── src/                        # shared config, prompts, LLM client, taxonomy
├── data/{raw,processed,external,output}/
└── notebooks/
```

## Getting Started

### Prerequisites

- Python >= 3.12
- Access to the Swissdox API
- On the UNIL cluster: SLURM, the `python/3.12.1` module, and a GPU partition for the LLM steps

### Installation

```bash
git clone https://github.com/ksrmaxime/accountability.git
cd accountability
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running on the UNIL cluster

```bash
bash launch_pipeline.sh              # full pipeline from download
bash launch_pipeline.sh step3 /path/to/step2_clean.parquet   # resume from a given step
```
