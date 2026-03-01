#!/usr/bin/env python3
"""
Evaluate a submission against a solution file using Citation-level Macro F1.

Usage:
    python evaluation/evaluate.py submission.csv
    python evaluation/evaluate.py submission.csv --split train
    python evaluation/evaluate.py submission.csv --solution path/to/solution.csv
    python evaluation/evaluate.py submission.csv -v
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class ParticipantVisibleError(Exception):
    pass


_WS_RE = re.compile(r"\s+")


def _canonicalize_citation(c: str) -> str:
    return _WS_RE.sub(" ", c.strip())


def _parse_citation_field(
    value: object,
    sep: str,
    max_items: int,
    max_chars: int,
) -> Set[str]:
    if pd.isna(value):
        return set()
    if not isinstance(value, str):
        value = str(value)
    if len(value) > max_chars:
        raise ParticipantVisibleError(
            f"predicted_citations field too long ({len(value)} chars). "
            f"Please limit to <= {max_chars} characters per query."
        )
    parts = value.split(sep)
    if len(parts) > max_items:
        raise ParticipantVisibleError(
            f"Too many citations in a single row ({len(parts)}). "
            f"Please limit to <= {max_items} citations per query."
        )
    return {canon for p in parts if (canon := _canonicalize_citation(p))}


def _f1_for_sets(pred: Set[str], gold: Set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    citation_separator: str = ";",
    max_citations_per_row: int = 200,
    max_chars_per_row: int = 10_000,
) -> float:
    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError(f"Solution is missing id column '{row_id_column_name}'.")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission is missing id column '{row_id_column_name}'.")

    solution = solution.set_index(row_id_column_name)
    submission = submission.set_index(row_id_column_name)

    missing_ids = set(solution.index) - set(submission.index)
    if missing_ids:
        raise ParticipantVisibleError(
            f"Submission is missing {len(missing_ids)} query IDs: {sorted(missing_ids)[:5]}..."
        )

    submission = submission.loc[solution.index]

    if len(submission.columns) != 1:
        raise ParticipantVisibleError(
            f"Submission must have exactly 1 prediction column (found {len(submission.columns)})."
        )

    pred_col = submission.columns[0]

    if "gold_citations" in solution.columns:
        gold_col = "gold_citations"
    elif pred_col in solution.columns:
        gold_col = pred_col
    elif len(solution.columns) == 1:
        gold_col = solution.columns[0]
    else:
        raise ParticipantVisibleError(
            "Solution file must have a 'gold_citations' column or match the submission column name."
        )

    f1s: List[float] = []
    for g, p in zip(solution[gold_col].tolist(), submission[pred_col].tolist()):
        gold_set = _parse_citation_field(g, citation_separator, max_citations_per_row, max_chars_per_row)
        pred_set = _parse_citation_field(p, citation_separator, max_citations_per_row, max_chars_per_row)
        f1s.append(_f1_for_sets(pred_set, gold_set))

    if not f1s:
        raise ParticipantVisibleError("No rows to score.")

    result = float(sum(f1s) / len(f1s))
    if not math.isfinite(result):
        raise ParticipantVisibleError("Score is not finite. Please check submission format.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a submission file using Citation-level Macro F1."
    )
    parser.add_argument("submission", type=Path, help="Path to submission CSV")
    parser.add_argument("--split", choices=["train", "val"], default="val",
                        help="Data split to evaluate against (default: val)")
    parser.add_argument("--solution", "-s", type=Path, dest="solution_path",
                        help="Path to a custom solution CSV (overrides --split)")
    parser.add_argument("--row-id", "-r", type=str, default="query_id",
                        help="Name of the row ID column (default: query_id)")
    parser.add_argument("--separator", "-sep", type=str, default=";",
                        help="Citation separator character (default: ;)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-query F1 scores")

    args = parser.parse_args()
    solution_path = args.solution_path or DEFAULT_DATA_DIR / f"{args.split}.csv"

    if not args.submission.exists():
        print(f"Error: Submission file not found: {args.submission}", file=sys.stderr)
        sys.exit(1)
    if not solution_path.exists():
        print(f"Error: Solution file not found: {solution_path}", file=sys.stderr)
        sys.exit(1)

    try:
        submission_df = pd.read_csv(args.submission)
        solution_df = pd.read_csv(solution_path)
    except Exception as e:
        print(f"Error reading CSV files: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Submission : {args.submission} ({len(submission_df)} rows)")
    print(f"Solution   : {solution_path} ({len(solution_df)} rows)")
    print()

    try:
        result = score(
            solution=solution_df,
            submission=submission_df,
            row_id_column_name=args.row_id,
            citation_separator=args.separator,
        )
        print(f"Macro F1 Score: {result:.6f}")

        if args.verbose:
            print("\nPer-query F1 scores:")
            solution_df = solution_df.set_index(args.row_id)
            submission_df = submission_df.set_index(args.row_id)
            submission_df = submission_df.loc[solution_df.index]
            pred_col = submission_df.columns[0]
            gold_col = "gold_citations" if "gold_citations" in solution_df.columns else solution_df.columns[0]
            for idx in solution_df.index:
                gold_set = _parse_citation_field(solution_df.loc[idx, gold_col], args.separator, 200, 10_000)
                pred_set = _parse_citation_field(submission_df.loc[idx, pred_col], args.separator, 200, 10_000)
                f1 = _f1_for_sets(pred_set, gold_set)
                print(f"  {idx}: F1={f1:.4f}  (pred={len(pred_set)}, gold={len(gold_set)})")

    except ParticipantVisibleError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error computing score: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()