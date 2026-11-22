"""Download yElo ratings from Tennis Abstract.

Scrapes the yElo leaderboard page and saves as CSV.

Usage:
    python scripts/download_elo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import csv
import urllib.request
import html

from config import DATA_DIR

EXTERNAL_DIR = DATA_DIR / "external"
ELO_URL = "https://tennisabstract.com/reports/atp_season_yelo_ratings.html"
ELO_FILE = EXTERNAL_DIR / "atp_yelo_current.csv"


def download_elo() -> None:
    """Scrape yElo ratings from Tennis Abstract."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    print("  [download] yElo ratings ...", end=" ")
    try:
        req = urllib.request.Request(ELO_URL, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode("utf-8")
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Parse table rows from HTML
    # The data is in a simple HTML table with align attributes
    rows = []
    # Find all table rows with player data
    # Pattern: <td align="right">RANK</td><td align="left"><a>NAME</a></td><td align="right">W</td><td align="right">L</td><td align="right">ELO</td>
    pattern = re.compile(
        r'<td[^>]*>(\d+)</td>\s*<td[^>]*><a[^>]*>([^<]+)</a></td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>([\d.]+)</td>'
    )

    for match in pattern.finditer(html_content):
        rank, name, wins, losses, elo = match.groups()
        # Decode HTML entities (e.g., &nbsp; -> space)
        name = html.unescape(name).replace('\xa0', ' ').strip()
        rows.append({
            "rank": int(rank),
            "player_name": name,
            "wins": int(wins),
            "losses": int(losses),
            "yelo": float(elo),
        })

    if not rows:
        print("  WARNING: No rows parsed. HTML structure may have changed.")
        return

    # Write CSV
    with open(ELO_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "player_name", "wins", "losses", "yelo"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} players to {ELO_FILE.name}")


def main() -> None:
    print("=== Downloading yElo Ratings (Tennis Abstract) ===\n")
    download_elo()
    print("\nDone.")


if __name__ == "__main__":
    main()
