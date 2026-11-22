"""Download ATP match data from Sackmann archive.

Downloads atp_matches_YYYY.csv files from the GitHub archive:
https://github.com/Aneeshers/tennis-sackmann-archive

Usage:
    python scripts/download_atp_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request
import csv
import io

from config import DATA_DIR

EXTERNAL_DIR = DATA_DIR / "external"
BASE_URL = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp"
YEARS = list(range(2010, 2025))


def download_file(url: str) -> str:
    """Download a file and return its content as string."""
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def download_atp_matches() -> None:
    """Download ATP match files for all years."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        filename = f"atp_matches_{year}.csv"
        url = f"{BASE_URL}/{filename}"
        dest = EXTERNAL_DIR / filename

        if dest.exists():
            print(f"  [skip] {filename} already exists")
            continue

        print(f"  [download] {filename} ...", end=" ")
        try:
            content = download_file(url)
            dest.write_text(content, encoding="utf-8")
            print(f"OK ({len(content):,} bytes)")
        except Exception as e:
            print(f"ERROR: {e}")


def main() -> None:
    print("=== Downloading ATP Match Data (Sackmann Archive) ===\n")
    download_atp_matches()
    print(f"\nDone. Files saved to {EXTERNAL_DIR}")


if __name__ == "__main__":
    main()
