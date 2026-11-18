# download_merge_years.py
"""
Regroupe les téléchargements annuels (sbatch_download_yearly.sh) en un
seul swissdox_articles_lead.parquet, identique au format d'un download classique,
pour que le step 2 n'ait rien à changer.

- Vérifie que TOUTES les années de la période (chain_info.txt) sont présentes ;
  sinon s'arrête en listant les années manquantes (rien n'est écrit).
- Écrit année par année (pyarrow ParquetWriter) : la mémoire utilisée reste
  celle d'une seule année, quelle que soit la taille totale.
- Types harmonisés entre années (une colonne vide une année ne casse pas le schéma).
- Écrit aussi articles_per_year.csv (contrôle rapide).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FILE = "swissdox_articles_lead.parquet"
DATE_COLS = {"pubtime"}
INT_COLS = {"char_count"}


def read_chain_info(chain_dir: Path) -> tuple[int, int]:
    info = dict(
        line.strip().split("=", 1)
        for line in (chain_dir / "chain_info.txt").read_text().splitlines() if "=" in line
    )
    return int(info["start_year"]), int(info["end_year"])


def normalize(df: pd.DataFrame, columns: list[str]) -> pa.Table:
    df = df.reindex(columns=columns)
    fields = []
    for c in columns:
        if c in DATE_COLS:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
            fields.append(pa.field(c, pa.date32()))
        elif c in INT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            fields.append(pa.field(c, pa.int64()))
        else:
            df[c] = df[c].astype("string")
            fields.append(pa.field(c, pa.string()))
    return pa.Table.from_pandas(df, schema=pa.schema(fields), preserve_index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain_dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    chain_dir, outdir = Path(args.chain_dir), Path(args.outdir)
    start, end = read_chain_info(chain_dir)
    years = list(range(start, end + 1))
    files = {y: chain_dir / f"year_{y}" / FILE for y in years}
    missing = [y for y, f in files.items() if not f.exists()]
    if missing:
        print(f"[ERROR] Années manquantes dans {chain_dir} : {missing}", file=sys.stderr)
        print("        Relancer la chaîne à partir de la 1re année manquante :", file=sys.stderr)
        print(f"        sbatch sbatch_download_yearly.sh {missing[0]} {end} {chain_dir}", file=sys.stderr)
        return 1

    # Colonnes : celles de la 1re année, plus toute colonne apparue ensuite
    columns: list[str] = []
    for f in files.values():
        for c in pq.read_schema(f).names:
            if c not in columns:
                columns.append(c)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / FILE
    tmp = outdir / (FILE + ".partial")
    writer = None
    counts = []
    for y in years:
        df = pd.read_parquet(files[y])
        table = normalize(df, columns)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table)
        counts.append({"year": y, "articles": len(df)})
        print(f"[merge] {y}: {len(df):,} articles", flush=True)
        del df, table
    writer.close()
    tmp.rename(out)   # le fichier final n'apparaît que si tout a réussi

    summary = pd.DataFrame(counts)
    summary.to_csv(outdir / "articles_per_year.csv", index=False)
    print(f"[merge] Total : {summary.articles.sum():,} articles, {len(years)} années → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
