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
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-results", type=int, default=500000)
    ap.add_argument("--outdir", default="data/input")
    ap.add_argument("--resume-query-id", default=None,
                    help="Skip submission and resume polling an existing Swissdox queryId")
    ap.add_argument("--test", action="store_true", help="Swissdox test mode (if supported)")
    args = ap.parse_args()

    languages = ["de", "fr"]
    sources = ["NZZO", "NNTA", "NNHEU", "ZWSO", "TPS", "NZZ", "TA", "ZWAO", "TPSO", "HEU", "ZWAS", "NZZS", "ZWAI"]

    query_name = f"BuerokratieVerwaltung_{datetime.now():%Y%m%d_%H%M%S}"
    comment = "THESIS pipeline: query -> clean -> article lead extraction"
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
        resume_query_id=args.resume_query_id,
        test=args.test,
    )

    print("[DONE] Outputs:")
    for k, p in out_paths.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
