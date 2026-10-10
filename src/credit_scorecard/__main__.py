"""Command-line entry point: ``python -m credit_scorecard``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .data import CATEGORICAL, load_taiwan
from .pipeline import develop, report
from .scorecard import ScalingParams


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="credit_scorecard",
        description="Develop a credit scorecard: binning, selection, scaling, validation.",
    )
    p.add_argument("data", nargs="?", type=Path,
                   help="CSV to model; omit to use the Taiwan default panel")
    p.add_argument("--target", default="default", help="name of the 0/1 target column")
    p.add_argument("--categorical", nargs="*", default=None,
                   help="columns to treat as categorical")
    p.add_argument("--base-score", type=float, default=600.0)
    p.add_argument("--base-odds", type=float, default=50.0)
    p.add_argument("--pdo", type=float, default=20.0, help="points to double the odds")
    p.add_argument("--min-iv", type=float, default=0.02)
    p.add_argument("--max-bins", type=int, default=6)
    p.add_argument("--corr-threshold", type=float, default=0.8)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protected", action="store_true",
                   help="include SEX and MARRIAGE (unlawful to score on; analysis only)")
    p.add_argument("-o", "--output", type=Path, help="write the points table here as CSV")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.data:
        df = pd.read_csv(args.data)
        if args.target not in df.columns:
            print(f"error: target {args.target!r} not in {list(df.columns)[:10]}", file=sys.stderr)
            return 2
        y = df[args.target].astype(int)
        X = df.drop(columns=[args.target])
        categorical = tuple(args.categorical or ())
    else:
        X, y = load_taiwan(protected=args.protected)
        categorical = tuple(args.categorical) if args.categorical is not None else CATEGORICAL

    result = develop(
        X, y,
        categorical=categorical,
        test_size=args.test_size,
        min_iv=args.min_iv,
        corr_threshold=args.corr_threshold,
        max_bins=args.max_bins,
        scaling=ScalingParams(args.base_score, args.base_odds, args.pdo),
        seed=args.seed,
    )
    print(report(result))

    if args.output:
        result.card.points_table.to_csv(args.output, index=False)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
