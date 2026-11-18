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
| download | `scripts/download.py` (+ `scripts/download_merge_years.py`) | Fetches raw articles from Swissdox (single query or one query per year) |
| 2 | `scripts/step2_clean_targets.py` (+ `src/step2_targets.py`) | Dedupes articles, finds every target (department, unit, agency, councillor) with the download keywords and SWISSDOX_REPO cleaning rules, one row per unique (article, target) + target metadata |
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
├── sbatch_download.sh / sbatch_download_yearly.sh / sbatch_download_merge_years.sh
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

### Download (Swissdox)

The download works like the one in SWISSDOX_REPO:

- **Query keywords**: all of them live in `src/keywords_list.py`: departments
  and federal units (`FEDERAL_UNITS`), independent agencies
  (`INDEPENDENT_AGENCIES`), both current + OLD, and federal councillors
  (`COUNCILLORS`). Every German declension and every French case/accent
  variant is written out, since Swissdox is case- and accent-sensitive.
  Every term with an apostrophe is also sent with the other apostrophe (' / ’).
  Step 2 does **not** use these lists (it matches via `src/target_taxonomy.py`).
- **Single query**: `sbatch sbatch_download.sh` (period set by `START`/`END`
  at the top of the script). If a job dies after submitting, resume the same
  query without resubmitting it: `sbatch sbatch_download.sh <QUERY_ID>`
  (the id is printed as `[Swissdox] queryId=...` in the job's `.out`).
- **Long periods (recommended)**: `sbatch sbatch_download_yearly.sh 2000 2025`
  runs one Swissdox query per year, strictly one after another, in
  `data/input/download_years_chain<ID>/year_YYYY/`. After the last year,
  `sbatch_download_merge_years.sh` merges them (plus `articles_per_year.csv`).
  To resume after a failed year:
  `sbatch sbatch_download_yearly.sh <year> 2025 <chain dir>` (finished years are skipped).
- Transient Swissdox errors (429/5xx, network) are retried with backoff; a
  progress line is logged every 10 minutes while waiting.
- Output: `data/input/download_job<ID>/swissdox_articles_lead.{parquet,csv}`.
  `data/input/.last_download` then points to it and `sbatch_step2.sh` is
  submitted automatically.


### Step 2 (targets)

`scripts/step2_clean_targets.py`, rules in `src/step2_targets.py`. Same
keywords as the download (`src/keywords_list.py`) and same cleaning rules as
SWISSDOX_REPO's sentence cutting, but the output stays at article level
(sentences are only used to judge each match in its context):

- photo credits ("Name / VBS", "(Bild: ...)", "© ...") removed and captions
  skipped before matching; sentences > 1000 chars or with an HTML entity skipped;
- full names case-insensitive, abbreviations case-sensitive (only the listed
  spellings: "SECO"/"Seco", never "seco"), whole words, longest match first;
- `STEP2_IGNORED_KEYWORDS`: terms that cannot identify one target
  ("Generalsekretariat", "Staatssekretariat", army units, ...);
- `STEP2_ABBREV_RULES`: ambiguous abbreviations kept only in a matching
  sentence context (BAK Economics, BSV Bern, "Res publica", Washington DC, ...);
- `STEP2_DATE_RULES` / `STEP2_TRANSFER_DATES`: terms shared by two targets and
  targets that changed department/status (ETH domain DFI -> DEFR in 2013,
  METAS office -> institute in 2013, private insurance office DFJP -> DFF in 2003);
- councillors kept only during their term of office.

One row per unique (article_id, keyword). Existing targets keep their
historical `keyword` label (GOLD files join on it). Columns added in front:
`matched_alias` (most explicit alias found, reference form), `keyword_abbrev`,
`keyword_mentions`, `target_type`, `target_status` (current/OLD), `parent_dept`
(historical format, empty for agencies), `department` + `department_en`
(every type), `target_name_{fr,de,en}`, `target_abbrev_{fr,de,en}`.

Output: `data/processed/step2_job<ID>/` with `step2_clean.{parquet,csv}`,
`keywords_reference.csv` (one row per target: labels, department, names and
abbreviations FR/DE/EN, validity dates, context rules, every searched
spelling) and `step2_target_counts.csv`. `data/processed/.last_step2` points to
the parquet; step 3 is submitted automatically.
