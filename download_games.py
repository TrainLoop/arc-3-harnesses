"""
Download all ARC-AGI-3 games from the API and cache them locally in dataset/.

Usage:
    pip install arc-agi
    python download_games.py

    # Or with a personal API key (gets access to more games):
    ARC_API_KEY=your-key python download_games.py

Games are saved as:
    dataset/
        index.json              # master index of all games
        games/
            {game_id}/
                metadata.json   # game metadata (id, title, tags, baseline_actions, etc.)
                source.py       # game source code (Python, uses arcengine)
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests

BASE_URL = "https://three.arcprize.org"
DATASET_DIR = Path(__file__).parent / "dataset"
GAMES_DIR = DATASET_DIR / "games"


def get_api_key() -> str:
    """Get API key from env or fetch anonymous one."""
    key = os.environ.get("ARC_API_KEY", "").strip()
    if key:
        print(f"Using provided API key: {key[:8]}...")
        return key

    print("No ARC_API_KEY set, fetching anonymous key...")
    resp = requests.get(f"{BASE_URL}/api/games/anonkey", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    key = data if isinstance(data, str) else data.get("key", data.get("api_key", ""))
    print(f"Got anonymous key: {key[:8]}...")
    return key


def list_games(api_key: str) -> list[dict]:
    """Fetch list of available games from API."""
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    resp = requests.get(f"{BASE_URL}/api/games", headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def download_game_source(api_key: str, game_id: str, version: str | None = None) -> str | None:
    """Download game source code."""
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}

    # Try with version suffix if available
    if version:
        url = f"{BASE_URL}/api/games/{game_id}-{version}/source"
    else:
        url = f"{BASE_URL}/api/games/{game_id}/source"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.HTTPError as e:
        if version:
            # Retry without version
            try:
                url2 = f"{BASE_URL}/api/games/{game_id}/source"
                resp2 = requests.get(url2, headers=headers, timeout=15)
                resp2.raise_for_status()
                return resp2.text
            except Exception:
                pass
        print(f"  WARNING: Could not download source for {game_id}: {e}")
        return None


def download_game_metadata(api_key: str, game_id: str) -> dict | None:
    """Fetch detailed metadata for a single game."""
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    try:
        resp = requests.get(f"{BASE_URL}/api/games/{game_id}", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  WARNING: Could not fetch metadata for {game_id}: {e}")
        return None


def main():
    api_key = get_api_key()

    # List all available games
    print("\nFetching game list...")
    games = list_games(api_key)
    print(f"Found {len(games)} games.\n")

    if not games:
        print("No games found. You may need a registered API key.")
        print("Register at https://three.arcprize.org")
        sys.exit(1)

    GAMES_DIR.mkdir(parents=True, exist_ok=True)

    index = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": BASE_URL,
        "total_games": len(games),
        "games": [],
    }

    for i, game_data in enumerate(games):
        game_id_full = game_data.get("game_id", "")
        title = game_data.get("title", game_id_full)

        # Parse game_id and version (format: "xx00-hexversion" or just "xx00")
        if "-" in game_id_full:
            game_id = game_id_full.split("-")[0]
            version = game_id_full.split("-", 1)[1]
        else:
            game_id = game_id_full
            version = None

        print(f"[{i+1}/{len(games)}] Downloading {game_id} ({title})...")

        game_dir = GAMES_DIR / game_id
        game_dir.mkdir(parents=True, exist_ok=True)

        # Fetch detailed metadata
        detailed_meta = download_game_metadata(api_key, game_id)
        metadata = detailed_meta if detailed_meta else game_data
        metadata["date_downloaded"] = datetime.now(timezone.utc).isoformat()

        # Save metadata
        meta_path = game_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Download source
        source = download_game_source(api_key, game_id, version)
        if source:
            source_path = game_dir / "source.py"
            source_path.write_text(source, encoding="utf-8")
            print(f"  -> Saved metadata + source to dataset/games/{game_id}/")
        else:
            print(f"  -> Saved metadata only to dataset/games/{game_id}/")

        index["games"].append({
            "game_id": game_id,
            "game_id_full": game_id_full,
            "title": title,
            "tags": game_data.get("tags", []),
            "baseline_actions": game_data.get("baseline_actions", []),
            "has_source": source is not None,
        })

    # Save master index
    index_path = DATASET_DIR / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"\nDone! Downloaded {len(games)} games to {DATASET_DIR}/")
    print(f"Master index: {index_path}")

    # Summary
    with_source = sum(1 for g in index["games"] if g["has_source"])
    print(f"  {with_source}/{len(games)} games have source code")
    print(f"  {len(games) - with_source}/{len(games)} games have metadata only")


if __name__ == "__main__":
    main()
