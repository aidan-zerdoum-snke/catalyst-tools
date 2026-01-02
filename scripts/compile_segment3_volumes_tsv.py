#!/usr/bin/env python3
"""
Compile Segment_3 metrics from multiple TSV files into a single CSV.

Usage:
  python compile_segment3_volumes_tsv.py "C:\\path\\to\\folder" \
      --output segment3_volumes_compiled.csv --recursive

Notes:
  - Extracts the "Segment_3" row per file.
  - Captures: volume_cm3, minimum, maximum, mean, std_dev.
  - Filename digits are used as the case number (e.g., '007.tsv' → case 7).
  - Files missing Segment_3 or required columns are skipped with a warning.
"""

import argparse
import csv
import os
import re
import sys
from typing import List, Dict, Optional, Union

import pandas as pd


def find_tsv_files(root: str, recursive: bool = False) -> List[str]:
    tsv_files: List[str] = []

    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".tsv"):
                    tsv_files.append(os.path.join(dirpath, fn))
    else:
        try:
            for fn in os.listdir(root):
                if fn.lower().endswith(".tsv"):
                    tsv_files.append(os.path.join(root, fn))
        except FileNotFoundError:
            print(f"❌ Folder not found: {root}", file=sys.stderr)
            return []

    return sorted(tsv_files)


def robust_read_tsv(path: str) -> Optional[pd.DataFrame]:
    """Try tab-separated first, then flexible whitespace."""
    try:
        return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8", engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=r"\s+", dtype=str, encoding="utf-8", engine="python")
        except Exception:
            return None


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Find first column matching exact name or regex (case-insensitive)."""
    # Exact matches first
    for p in patterns:
        for c in df.columns:
            if c == p:
                return c

    # Regex fallback
    for c in df.columns:
        for p in patterns:
            if re.search(p, c, flags=re.IGNORECASE):
                return c

    return None


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def extract_case_id(path: str) -> Union[int, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = re.sub(r"\D", "", stem)

    if digits.isdigit():
        return int(digits.lstrip("0") or "0")

    return stem


def extract_segment3_record(path: str) -> Optional[Dict]:
    df = robust_read_tsv(path)
    if df is None or df.empty:
        print(f"⚠️  Skipping unreadable or empty file: {path}", file=sys.stderr)
        return None

    df = normalize(df)

    if "Segment" not in df.columns:
        print(f"⚠️  Skipping (no 'Segment' column): {path}", file=sys.stderr)
        return None

    seg3_mask = (
        df["Segment"]
        .astype(str)
        .str.lower()
        .str.contains(r"segment\s*_?\s*3")
    )

    seg3_rows = df[seg3_mask]
    if seg3_rows.empty:
        print(f"⚠️  Skipping (no Segment_3 row): {path}", file=sys.stderr)
        return None

    row = seg3_rows.iloc[0]

    vol_col = find_col(df, [r"^Volume\s*cm3$", r"\bvolume\b", r"\bcm3\b"])
    min_col = find_col(df, [r"^Minimum$", r"\bmin\b"])
    max_col = find_col(df, [r"^Maximum$", r"\bmax\b"])
    mean_col = find_col(df, [r"^Mean$", r"\bmean\b"])
    std_col = find_col(df, [r"^Standard\s*deviation$", r"\bstd\b", r"\bstdev\b"])

    return {
        "case": extract_case_id(path),
        "file": os.path.relpath(path),
        "segment": str(row.get("Segment", "")).strip(),
        "volume_cm3": to_float(row.get(vol_col)) if vol_col else None,
        "minimum": to_float(row.get(min_col)) if min_col else None,
        "maximum": to_float(row.get(max_col)) if max_col else None,
        "mean": to_float(row.get(mean_col)) if mean_col else None,
        "std_dev": to_float(row.get(std_col)) if std_col else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Segment_3 metrics from TSV files into a single CSV."
    )
    parser.add_argument("folder", help="Folder containing TSV files")
    parser.add_argument(
        "--output", "-o",
        default="segment3_volumes_compiled.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subfolders recursively"
    )

    args = parser.parse_args()

    tsv_files = find_tsv_files(args.folder, recursive=args.recursive)
    if not tsv_files:
        print(
            f"❌ No .tsv files found in {args.folder} "
            f"(recursive={args.recursive}).",
            file=sys.stderr
        )
        sys.exit(1)

    records: List[Dict] = []
    for path in tsv_files:
        rec = extract_segment3_record(path)
        if rec:
            records.append(rec)

    if not records:
        print("❌ No Segment_3 rows found in the provided files.", file=sys.stderr)
        sys.exit(2)

    # Sort numeric cases first, then non-numeric
    records.sort(
        key=lambda r: (
            r["case"] if isinstance(r["case"], int) else float("inf"),
            str(r["case"]),
        )
    )

    fieldnames = [
        "case",
        "file",
        "segment",
        "volume_cm3",
        "minimum",
        "maximum",
        "mean",
        "std_dev",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
