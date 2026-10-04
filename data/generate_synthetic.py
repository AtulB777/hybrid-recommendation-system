"""CLI: python -m data.generate_synthetic --users 500 --items 400 --seed 42"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.services.ingestion import DataBundle, write_csv_bundle
from app.services.synthetic_data import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset as CSV files.")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=get_settings().raw_data_dir)
    args = parser.parse_args()

    data = generate_synthetic_dataset(args.users, args.items, seed=args.seed)
    write_csv_bundle(DataBundle(**data), args.out)
    print(
        f"Wrote {len(data['users'])} users, {len(data['items'])} items, "
        f"{len(data['interactions'])} interactions to {args.out}"
    )


if __name__ == "__main__":
    main()
