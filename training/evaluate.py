"""Compare all models:  python -m training.evaluate [--k 5 10 20] [--source csv|db]

Uses the weights stored in the trained artifact when present, otherwise the defaults.
"""
from __future__ import annotations

import argparse

from app.config import get_settings
from app.database.session import make_engine
from app.evaluation.runner import format_markdown_table, run_comparison
from app.services.artifacts import load_bundle
from app.services.ingestion import load_bundle_from_db, load_csv_bundle, validate_bundle
from training.train import write_report


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--source", choices=["csv", "db"], default="csv")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--write", action="store_true", help="overwrite docs/evaluation_results.*")
    args = p.parse_args()

    data = validate_bundle(load_csv_bundle(s.raw_data_dir))[0] if args.source == "csv" \
        else load_bundle_from_db(make_engine(s.database_url))
    weights = load_bundle(s.artifact_path).hybrid_weights if s.artifact_path.exists() else None
    report = run_comparison(data, ks=args.k, hybrid_weights=weights, half_life_days=s.half_life_days, test_frac=args.test_frac)
    print(f"protocol: {report['protocol']}\n")
    for k in report["ks"]:
        print(f"K = {k}\n{format_markdown_table(report, k)}\n")
    if args.write:
        write_report(report, s.docs_dir, k=max(args.k) if 10 not in args.k else 10)


if __name__ == "__main__":
    main()
