#!/usr/bin/env python3
"""
LLM Harness for ARC-AGI-3 Games (Remote API version)
Plays games through the ARC API so recordings sync to the website.
Uses the same BFS solver but executes via HTTP instead of locally.
"""

import sys
import os
import json
import time
import logging
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from arcengine import GameAction

# Import the local solver
from harness import GridSolver, load_game, ACTION_NAMES, build_state_description, LLMSolver

BASE_URL = "https://three.arcprize.org"
GAME_ACTION_MAP = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
}


class ArcAPIClient:
    """Simple client for the ARC-AGI-3 API."""

    def __init__(self, api_key: str, base_url: str = BASE_URL):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Accept": "application/json",
        })

    def open_scorecard(self, tags=None) -> str:
        """Open a new scorecard, returns card_id."""
        payload = {"tags": tags or ["llm-harness"]}
        resp = self.session.post(
            f"{self.base_url}/api/scorecard/open",
            json=payload,
            headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, str) else data.get("card_id", data.get("id", ""))

    def close_scorecard(self, card_id: str):
        """Close a scorecard."""
        resp = self.session.post(
            f"{self.base_url}/api/scorecard/close",
            json={"card_id": card_id},
            headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def reset(self, game_id: str, card_id: str, guid: Optional[str] = None) -> dict:
        """Reset a game, returns frame data."""
        payload = {"card_id": card_id, "game_id": game_id}
        if guid:
            payload["guid"] = guid
        resp = self.session.post(
            f"{self.base_url}/api/cmd/RESET",
            json=payload,
            headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def action(self, game_id: str, guid: str, action: GameAction,
               reasoning: Optional[str] = None) -> dict:
        """Send an action, returns frame data."""
        action_name = f"ACTION{action.value}"
        payload = {"game_id": game_id, "guid": guid}
        if reasoning:
            payload["reasoning"] = reasoning
        resp = self.session.post(
            f"{self.base_url}/api/cmd/{action_name}",
            json=payload,
            headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


def get_api_key() -> str:
    """Get API key from env or fetch anonymous one."""
    key = os.environ.get("ARC_API_KEY", "").strip()
    if key:
        return key
    print("No ARC_API_KEY set, fetching anonymous key...")
    resp = requests.get(f"{BASE_URL}/api/games/anonkey", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    key = data if isinstance(data, str) else data.get("key", data.get("api_key", ""))
    print(f"Got anonymous key: {key[:8]}...")
    return key


def main():
    game_id = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    base_dir = Path(__file__).parent
    game_dir = base_dir / "dataset" / "games" / game_id
    source_path = game_dir / "source.py"
    meta_path = game_dir / "metadata.json"

    if not game_dir.exists():
        print(f"Error: Game not found: {game_dir}")
        sys.exit(1)

    metadata = json.loads(meta_path.read_text())
    game_full_id = metadata.get("game_id", game_id)
    source_code = source_path.read_text()

    print(f"ARC-AGI-3 LLM Harness (Remote API)")
    print(f"{'='*60}")
    print(f"Game: {game_full_id}")

    # Get API key and set up client
    api_key = get_api_key()
    client = ArcAPIClient(api_key)

    # Open scorecard
    print("Opening scorecard...")
    card_id = client.open_scorecard(tags=["llm-harness", game_id])
    print(f"Scorecard: {card_id}")

    # Load game locally for solver (read-only, just for planning)
    local_game = load_game(str(game_dir))
    from arcengine import ActionInput
    local_game.perform_action(ActionInput(id=GameAction.RESET))

    # Reset game via API
    print("Resetting game via API...")
    frame = client.reset(game_full_id, card_id)
    guid = frame.get("guid")
    win_levels = frame.get("win_levels", 7)
    print(f"GUID: {guid}")
    print(f"Levels: {win_levels}")
    print(f"State: {frame.get('state')}")

    total_actions = 0
    levels_solved = 0
    level_stats = []

    for level_idx in range(win_levels):
        print(f"\n{'='*60}")
        print(f"LEVEL {level_idx + 1}")
        level_start = time.time()
        level_actions = 0

        # Plan using local solver
        solver = GridSolver(local_game)
        actions = solver.solve_level()

        solver_type = "BFS"
        if actions is None:
            solver_type = "LLM"
            print("  BFS failed, trying LLM...")
            state_desc = build_state_description(local_game)
            llm = LLMSolver(source_code)
            actions = llm.solve(state_desc)

        if actions is None:
            print(f"  No solution found for level {level_idx + 1}")
            level_stats.append({"level": level_idx + 1, "solved": False, "actions": 0})
            break

        print(f"  [{solver_type}] Plan: {len(actions)} actions")

        # Execute via API
        prev_levels = frame.get("levels_completed", 0)
        level_solved = False

        for i, act in enumerate(actions):
            ga = GAME_ACTION_MAP.get(act)
            if ga is None:
                continue

            try:
                frame = client.action(game_full_id, guid, ga)
            except Exception as e:
                print(f"  API error at action {i}: {e}")
                break

            level_actions += 1
            total_actions += 1

            # Also step the local game to keep solver in sync
            local_game.perform_action(ActionInput(id=ga))

            cur_levels = frame.get("levels_completed", 0)
            state = frame.get("state", "NOT_FINISHED")

            if cur_levels > prev_levels:
                level_solved = True
                break
            if state == "WIN":
                level_solved = True
                break
            if state == "GAME_OVER":
                print(f"  GAME OVER at action {i+1}")
                break

        elapsed = time.time() - level_start
        status = "SOLVED" if level_solved else "FAILED"
        print(f"  >> Level {level_idx + 1}: {status} in {level_actions} actions ({elapsed:.2f}s)")

        level_stats.append({
            "level": level_idx + 1,
            "solved": level_solved,
            "actions": level_actions,
            "time": round(elapsed, 2),
        })

        if level_solved:
            levels_solved += 1
        else:
            break

        if frame.get("state") in ("WIN", "GAME_OVER"):
            break

    # Close scorecard
    print(f"\nClosing scorecard...")
    try:
        result = client.close_scorecard(card_id)
        print(f"Scorecard closed: {result}")
    except Exception as e:
        print(f"Error closing scorecard: {e}")

    # Save results
    results_path = base_dir / f"{game_id}_results.txt"
    with open(str(results_path), "w") as f:
        f.write(f"ARC-AGI-3 LLM Harness Results (Remote API)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Game: {game_full_id}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Scorecard: {card_id}\n")
        f.write(f"GUID: {guid}\n\n")
        f.write(f"Levels solved: {levels_solved} / {win_levels}\n")
        f.write(f"Total actions: {total_actions}\n\n")
        for stat in level_stats:
            s = "SOLVED" if stat["solved"] else "FAILED"
            f.write(f"  Level {stat['level']}: {s} | {stat['actions']} actions\n")

    print(f"\nResults: {results_path}")
    print(f"Summary: {levels_solved}/{win_levels} levels solved")
    print(f"\nView recording at: https://three.arcprize.org")


if __name__ == "__main__":
    main()
