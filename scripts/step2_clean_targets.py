"""
scripts/step2_clean_targets.py

STEP 2 of the accountability pipeline — turns the raw Swissdox download
(one row = one article) into the clean, LLM-ready dataset
(one row = one UNIQUE article x target combination).

Same keywords as the Swissdox query (src/keywords_list.py) and same cleaning
rules as SWISSDOX_REPO's sentence cutting, but the output stays at ARTICLE
level: sentences are only used to judge each match in its context. All the
matching rules live in src/step2_targets.py.

What this step does, in order:

  1. Drop duplicate articles: when two rows have the exact same (non-empty)
     title, keep only the one published first (earliest pubtime).
  2. Split each article into sentences (same splitter as SWISSDOX_REPO) and
     look for every department, unit, agency and councillor in each sentence:
       - photo credits removed from the sentence first ("Name / EPA",
         "(Bild: ...)", "© ...") and caption lines skipped, so a credit never
         creates a target ;
       - sentences > 1000 chars (tables) or containing an HTML entity skipped ;
       - full names case-insensitive, abbreviations case-SENSITIVE (only the
         spellings listed in keywords_list.py), whole words only ;
       - terms that cannot identify one target are ignored
         ("Generalsekretariat", "Staatssekretariat", army units, ...) ;
       - ambiguous abbreviations kept only if their sentence passes the
         context rule (DC needs a consular/EDA context, PSI a research
         context, ...) ;
       - terms shared by two targets, and targets that moved department or
         status, resolved with the article's publication date.
  3. One output row per (article, target). Every alias of the same target
     (FR/DE, abbreviation/full name, declensions, case variants) gives the
     same `keyword`; `matched_alias` is the most explicit mention found in
     the article (longest), EXACTLY as written in the text (e.g.
     "Bundesamtes für Gesundheit", "Secrétariat d'État à l'économie") so the
     step 3+ prompts anchor on the article's own wording; `keyword_mentions`
     counts how many times the target was matched in the article.
  4. Target metadata on every row: keyword_abbrev, target_type, target_status
     (current / OLD), parent_dept (historical format, as in the GOLD files),
     department + department_en (for every type, agencies included),
     target_name_{fr,de,en}, target_abbrev_{fr,de,en}.
  5. Federal Councillor rows dropped if the article was published outside
     that person's term of office.
  6. Safety net: one row per (article_id, keyword), earliest version kept.

Outputs (in --output_dir):
  step2_clean.parquet / .csv   the dataset
  keywords_reference.csv       one row per target record: labels, department,
                               names and abbreviations FR/DE/EN, validity
                               dates, context rules, every searched spelling
  step2_target_counts.csv      rows per target (quick sanity check)

Usage
-----
  python scripts/step2_clean_targets.py \\
      --input  data/input/download_job<ID>/swissdox_articles_lead.parquet \\
      --output_dir data/processed/step2_job<ID> \\
      --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.step2_targets import (  # noqa: E402
    FEDERAL_COUNCILLOR,
    STEP2_MAX_SENTENCE_CHARS,
    _HTML_ENTITY,
    _day,
    best_alias,
    build_pattern,
    legacy_targets_lost,
    reference_table,
    resolve_sentence,
    target_metadata,
)
from src.target_taxonomy import is_councillor_in_office  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Sentence splitting, credits and captions — copied from
# SWISSDOX_REPO/scripts/step2_sentence_cutting.py (keep the two in sync).
# ════════════════════════════════════════════════════════════════════════════
_PHOTO_LBL = r"(?:Bild|Bilder|Foto|Fotos|Photo|Photos|Image|Images|Illustration|Grafik|Karte)"
_LINE_LBL = r"(?:Bild|Bilder|Foto|Fotos|Photo|Photos|Image|Images|Illustration|Grafik|Karte|Quelle|Quellen|Source|Sources)"
_AGENCY = r"(?:EPA|Keystone|KEYSTONE|Keytone|Reuters|REUTERS|AFP|dpa|DPA|Imago|IMAGO|AP|Getty(?:\s+Images)?|freshfocus|Freshfocus)"
_W = r"[^\s/()]+"
_PAREN_CREDIT = re.compile(r"\(\s*" + _PHOTO_LBL + r"\s*:[^)]*(?:\)|$)", re.I)
_COPYRIGHT = re.compile(r"©.*$")
_AGENCY_CHAIN = re.compile(
    r"(?:" + _W + r"\s+){0,2}" + _W + r"\s*/\s*" + _AGENCY + r"(?:\s*/\s*" + _W + r")*"
    r"|" + _AGENCY + r"\s*/\s*" + _W + r"(?:\s*/\s*" + _W + r")*"
    r"|\bvia\s+" + _AGENCY + r"\b|\b" + _AGENCY + r"\s+via\b"
)
_LEAD_LABEL = re.compile(r"^[\s–—\-(]*" + _LINE_LBL + r"\s*:", re.I)


def strip_credits(sent: str) -> str:
    sent = _PAREN_CREDIT.sub(" ", sent)
    sent = _COPYRIGHT.sub(" ", sent)
    sent = _AGENCY_CHAIN.sub(" ", sent)
    return re.sub(r"\s+", " ", sent).strip()


def is_caption(sent: str) -> bool:
    core = sent.strip(" ()[]–—-.,;:➤■")
    if not core:
        return True
    n_words = len(core.split())
    if _LEAD_LABEL.match(sent) and n_words <= 15:
        return True
    if "/" in core and n_words <= 8 and not re.search(r"[.!?]$", sent.rstrip(" )")):
        return True
    return False


_PLACEHOLDER = '\x00'
_ABBREVS = sorted([
    'Prof', 'Dr', 'Hr', 'Fr', 'Mr', 'Mrs', 'Ms', 'Mme', 'M',
    'St', 'Kt', 'Art', 'Abs', 'Ziff', 'lit', 'Bst', 'Nr', 'No',
    'bzw', 'resp', 'etc', 'usw', 'ua', 'ca', 'vgl', 'ggf',
    'inkl', 'exkl', 'evtl', 'insb', 'sog', 'mind', 'max', 'min',
    'Co', 'Cie',
    'Jan', 'Feb', 'Mär', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez',
    'cf', 'vs', 'fig', 'Fig', 'Bd', 'Jg', 'ff', 'op',
], key=len, reverse=True)
_abbrev_re = re.compile(r'\b(' + '|'.join(re.escape(a) for a in _ABBREVS) + r')\.(\s)', re.IGNORECASE)
_ordinal_re = re.compile(r'(\d+)\.(\s)')
_initial_re = re.compile(r'\b([A-ZÜÄÖ])\.(\s+[A-ZÜÄÖÉÀÂÊ])')
_multidot_re = re.compile(r'\b([a-zA-Züäö]\.)+([a-zA-Züäö])\.(?=\s)')


def _protect_ordinal(m: re.Match) -> str:
    """'3. Juli' : not a sentence end — except a year ('... im Jahr 2014. Der ...')."""
    num = m.group(1)
    if len(num) == 4 and 1800 <= int(num) <= 2099:
        return m.group(0)
    return num + _PLACEHOLDER + m.group(2)


def _split_on_punctuation(t: str) -> list[str]:
    t = _multidot_re.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), t)
    t = _abbrev_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    t = _ordinal_re.sub(_protect_ordinal, t)
    t = _initial_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    parts = re.split(r'(?<=[.!?])\s+', t)
    return [p.replace(_PLACEHOLDER, '.').strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Line breaks are hard boundaries, then punctuation-aware splitting."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            sentences.extend(_split_on_punctuation(line))
    return sentences


# ════════════════════════════════════════════════════════════════════════════
# Article -> targets
# ════════════════════════════════════════════════════════════════════════════
def article_targets(text, lang, pubtime, pattern, stats: dict) -> list[tuple[str, str, int]]:
    """[(keyword, matched_alias, n_mentions)] for one article, sorted by keyword."""
    found: dict[str, dict[str, int]] = {}
    counts: dict[str, int] = {}
    day = _day(pubtime)
    offset = 0
    for sent in split_sentences(text if isinstance(text, str) else ""):
        start, offset = offset, offset + len(sent) + 1
        sent = strip_credits(sent)
        if not sent:
            continue
        if is_caption(sent):
            stats["captions"] += 1
            continue
        if len(sent) > STEP2_MAX_SENTENCE_CHARS:
            stats["too_long"] += 1
            continue
        if _HTML_ENTITY.search(sent):
            stats["html_entity"] += 1
            continue
        resolve_sentence(sent, pattern, lang, day, start, found, counts)
    return sorted((kw, best_alias(aliases), counts[kw]) for kw, aliases in found.items())


def extract_chunk(chunk: pd.DataFrame, text_col: str, lang_col: str, pubtime_col: str,
                  pattern) -> tuple[list[dict], dict]:
    stats = {"captions": 0, "too_long": 0, "html_entity": 0}
    rows = []
    for row in chunk.to_dict("records"):
        lang = row.get(lang_col)
        pubtime = row.get(pubtime_col)
        for keyword, alias, n in article_targets(row.get(text_col), lang, pubtime, pattern, stats):
            out = dict(row)
            out["keyword"] = keyword
            out["matched_alias"] = alias
            out["keyword_mentions"] = n
            out.update(target_metadata(keyword, pubtime, lang))
            rows.append(out)
    return rows, stats


# Parallelism: each worker compiles its own pattern once.
_WORKER_PATTERN = None
_WORKER_COLS = None


def _init_worker(cols):
    global _WORKER_PATTERN, _WORKER_COLS
    _WORKER_PATTERN = build_pattern()
    _WORKER_COLS = cols


def _extract_in_worker(chunk):
    return extract_chunk(chunk, *_WORKER_COLS, _WORKER_PATTERN)


# ════════════════════════════════════════════════════════════════════════════
# Article-level cleaning
# ════════════════════════════════════════════════════════════════════════════
def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def drop_duplicate_titles(df: pd.DataFrame, *, title_col: str = "title", pubtime_col: str = "pubtime") -> pd.DataFrame:
    """Keep one row per exact title (earliest pubtime wins). Rows with a
    missing/empty title are never considered duplicates of one another."""
    title = df[title_col].astype("string").str.strip()
    has_title = title.notna() & (title != "")

    with_title = df[has_title].copy()
    without_title = df[~has_title]

    with_title["_title_norm"] = title[has_title]
    with_title = with_title.sort_values(pubtime_col, na_position="last", kind="stable")
    before = len(with_title)
    with_title = with_title.drop_duplicates(subset="_title_norm", keep="first")
    with_title = with_title.drop(columns="_title_norm")
    dropped = before - len(with_title)

    out = pd.concat([with_title, without_title], ignore_index=False).sort_index()
    print(f"[dedup] {dropped:,} duplicate-title articles dropped "
          f"({before:,} -> {len(with_title):,} titled articles; "
          f"{len(without_title):,} untitled articles kept as-is)")
    return out


def filter_councillors_in_office(df: pd.DataFrame, *, pubtime_col: str) -> pd.DataFrame:
    is_councillor = df["target_type"] == FEDERAL_COUNCILLOR
    in_office = pd.Series(True, index=df.index)
    in_office[is_councillor] = [
        is_councillor_in_office(name, pubtime)
        for name, pubtime in zip(df.loc[is_councillor, "keyword"], df.loc[is_councillor, pubtime_col])
    ]
    dropped = int((~in_office).sum())
    print(f"[councillors] {dropped:,} rows dropped (article published outside the "
          f"named councillor's term of office)")
    return df[in_office].copy()


FRONT_COLS = [
    "article_id", "keyword", "matched_alias", "keyword_abbrev", "keyword_mentions",
    "target_type", "target_status", "parent_dept", "department", "department_en",
    "target_name_fr", "target_name_de", "target_name_en",
    "target_abbrev_fr", "target_abbrev_de", "target_abbrev_en",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Download output: parquet or csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--title_col", default="title")
    ap.add_argument("--lang_col", default="language")
    ap.add_argument("--pubtime_col", default="pubtime")
    ap.add_argument("--chunk_size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="Parallel processes (default: SLURM_CPUS_PER_TASK, else 1)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference table first: it documents exactly what this run searched for.
    ref = reference_table()
    ref.to_csv(out_dir / "keywords_reference.csv", index=False, encoding="utf-8-sig")
    print(f"[step2] keywords_reference.csv: {ref['keyword'].nunique()} targets, "
          f"{len(ref)} records, {int(ref['n_surfaces'].sum()):,} searched spellings")
    lost = legacy_targets_lost()
    if lost:
        print(f"[step2] NOTE: targets of the old taxonomy no longer in keywords_list.py: {lost}")

    print(f"[step2] Reading {in_path} ...")
    df = load(in_path)
    print(f"[step2] {len(df):,} articles loaded")

    df = drop_duplicate_titles(df, title_col=args.title_col, pubtime_col=args.pubtime_col)
    df = df.reset_index(drop=True)

    cols = (args.text_col, args.lang_col, args.pubtime_col)
    chunks = [df.iloc[s:s + args.chunk_size] for s in range(0, len(df), args.chunk_size)]
    workers = max(1, args.workers)
    print(f"[step2] {len(chunks)} chunks of {args.chunk_size} articles, {workers} process(es)", flush=True)

    rows: list[dict] = []
    stats = {"captions": 0, "too_long": 0, "html_entity": 0}
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker, initargs=(cols,)) as pool:
            results = pool.imap(_extract_in_worker, chunks)
            for i, (chunk_rows, chunk_stats) in enumerate(results, 1):
                rows.extend(chunk_rows)
                for k, v in chunk_stats.items():
                    stats[k] += v
                if i % 50 == 0 or i == len(chunks):
                    print(f"[step2] chunk {i}/{len(chunks)} -> {len(rows):,} rows", flush=True)
    else:
        pattern = build_pattern()
        for i, chunk in enumerate(chunks, 1):
            chunk_rows, chunk_stats = extract_chunk(chunk, *cols, pattern)
            rows.extend(chunk_rows)
            for k, v in chunk_stats.items():
                stats[k] += v
            if i % 50 == 0 or i == len(chunks):
                print(f"[step2] chunk {i}/{len(chunks)} -> {len(rows):,} rows", flush=True)

    print(f"[step2] sentences skipped: captions {stats['captions']:,} | "
          f"> {STEP2_MAX_SENTENCE_CHARS} chars {stats['too_long']:,} | "
          f"HTML entity {stats['html_entity']:,}")

    final = pd.DataFrame.from_records(rows)
    if final.empty:
        final = pd.DataFrame(columns=FRONT_COLS + list(df.columns))
    print(f"[explode] {len(df):,} articles -> {len(final):,} article-target rows "
          f"({final['keyword'].nunique()} distinct targets)")

    final = filter_councillors_in_office(final, pubtime_col=args.pubtime_col)

    # Safety net: a handful of Swissdox entries reuse the same article_id for a
    # live article re-titled after publication (title dedup does not catch it).
    before = len(final)
    final = final.sort_values(args.pubtime_col, na_position="last", kind="stable")
    final = final.drop_duplicates(subset=["article_id", "keyword"], keep="first")
    if before - len(final):
        print(f"[dedup] {before - len(final):,} rows dropped for reused article_id "
              f"(same article_id, same target; kept the earliest-published version)")
    final = final.sort_values(["article_id", "keyword"], kind="stable").reset_index(drop=True)

    other = [c for c in final.columns if c not in FRONT_COLS]
    final = final[[c for c in FRONT_COLS if c in final.columns] + other]

    out_parquet = out_dir / "step2_clean.parquet"
    out_csv = out_dir / "step2_clean.csv"
    final.to_parquet(out_parquet, index=False)
    final.to_csv(out_csv, index=False)

    counts = (final.groupby(["keyword", "target_type"]).size().rename("rows")
              .reset_index().sort_values("rows", ascending=False))
    counts.to_csv(out_dir / "step2_target_counts.csv", index=False)

    print(f"\n[step2] Final dataset: {len(final):,} article-target rows "
          f"({final['article_id'].nunique():,} distinct articles, "
          f"{final['keyword'].nunique():,} distinct targets)")
    print(final["target_type"].value_counts().to_string())
    print(f"[step2] Saved -> {out_parquet}")
    print(f"[step2] Saved -> {out_csv}")
    print(f"[step2] Saved -> {out_dir / 'keywords_reference.csv'}")
    print(f"[step2] Saved -> {out_dir / 'step2_target_counts.csv'}")

    # Pointer for the next step (same convention as before: data/processed/.last_step2)
    pointer = out_dir.parent / ".last_step2"
    pointer.write_text(str(out_parquet.resolve()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
