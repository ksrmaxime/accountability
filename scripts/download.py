from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datetime import datetime

import argparse

from src.download_src import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-results", type=int, default=1000000)
    ap.add_argument("--outdir", required=True,
                    help="Dossier propre à ce téléchargement (ex. data/input/download_job<ID>)")
    ap.add_argument("--test", action="store_true", help="Swissdox test mode (if supported)")
    ap.add_argument("--max-wait-hours", type=float, default=12.0, help="Max time to wait for Swissdox downloadUrl")
    ap.add_argument("--query-id", default=None,
                    help="Reprendre une requête Swissdox déjà soumise (queryId affiché dans le .out d'un job précédent) au lieu d'en soumettre une nouvelle")
    args = ap.parse_args()

    languages = ["de", "fr"]
    sources = ["NZZO", "NNTA", "NNHEU", "ZWSO", "TPS", "NZZ", "TA", "ZWAO", "TPSO", "HEU", "ZWAS", "NZZS", "ZWAI"]

    query_name = f"Accountability_{datetime.now():%Y%m%d_%H%M%S}"
    comment = "Accountability pipeline: query -> clean -> criticism/source/content-type/response classification"
    expiration_date = "2026-12-31"

    out_paths = run_pipeline(
        start_date=args.start,
        end_date=args.end,
        languages=languages,
        sources=sources,
        max_results=args.max_results,
        expiration_date=expiration_date,
        query_name=query_name,
        comment=comment,
        out_dir=Path(args.outdir),
        test=args.test,
        max_wait_s=int(args.max_wait_hours * 3600),
        query_id=args.query_id,
    )

    print("[DONE] Outputs:")
    for k, p in out_paths.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
