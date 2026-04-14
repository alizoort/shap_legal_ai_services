from __future__ import annotations

import argparse
import csv
from pathlib import Path

from services.legal_ai_service.assets.synthetic_dataset_source import (
    build_gold_records,
    build_training_records,
)


def export_datasets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_records(output_dir / "training.csv", build_training_records())
    _write_records(output_dir / "gold.csv", build_gold_records())


def _write_records(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "risk_label", "rationale", "category"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "text": record.text,
                    "risk_label": record.risk_label,
                    "rationale": record.rationale,
                    "category": record.category,
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deterministic synthetic datasets for the legal AI POC.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory that will receive training.csv and gold.csv",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    export_datasets(args.output_dir)


if __name__ == "__main__":
    main()
