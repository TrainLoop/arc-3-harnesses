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
from harness import GridSolver, load_game, ACTION_NAMES, ACTION_DELTAS, build_state_description, LLMSolver

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
               reasoning=None) -> dict:
        """Send an action, returns frame data."""
        action_name = f"ACTION{action.value}"
        payload = {"game_id": game_id, "guid": guid}
        if reasoning is not None:
            # Send as JSON string per remote_wrapper convention
            payload["reasoning"] = json.dumps(reasoning) if not isinstance(reasoning, str) else reasoning
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


class ReasoningLogger:
    """Generates reasoning logs for each action in the solver plan."""

    def __init__(self, game, solver, actions, level_idx):
        self.game = game
        self.solver = solver
        self.actions = actions
        self.level_idx = level_idx
        self.pos = solver.start_pos
        self.pw, self.ph = solver.pw, solver.ph

        # Analyze the plan to annotate each action with context
        self.annotations = self._annotate_plan()

    def _annotate_plan(self):
        """Pre-analyze the full plan to assign a phase/reason to each action."""
        s = self.solver
        goal = s.goals[0] if s.goals else None

        # Compute needed changes
        shape_steps = (goal["shape"] - s.shape) % s.n_shapes if goal else 0
        color_steps = (goal["color_idx"] - s.color_idx) % s.n_colors if goal else 0
        rot_steps = (goal["rotation_idx"] - s.rotation_idx) % 4 if goal else 0

        changer_info = []
        if shape_steps > 0 and "shape" in s.changers:
            changer_info.append(("shape", shape_steps, s.changers["shape"][0]))
        if color_steps > 0 and "color" in s.changers:
            changer_info.append(("color", color_steps, s.changers["color"][0]))
        if rot_steps > 0 and "rotation" in s.changers:
            changer_info.append(("rotation", rot_steps, s.changers["rotation"][0]))

        # Simulate the path to identify phases
        annotations = []
        pos = self.pos
        coll_positions = set()
        for cx, cy in s.collectibles:
            cp = s._collection_pos(cx, cy)
            if cp:
                coll_positions.add(cp)

        changer_positions = set()
        for _, _, cpos in changer_info:
            changer_positions.add(cpos)

        visits_remaining = {ct: n for ct, n, _ in changer_info}
        current_target = None
        phase = "navigate"

        for i, act in enumerate(self.actions):
            dc, dr = ACTION_DELTAS[act]
            nx, ny = pos[0] + dc * self.pw, pos[1] + dr * self.ph
            if s.in_bounds(nx, ny) and not s.is_blocked(nx, ny):
                if (nx, ny) in s.push_teleports:
                    dest = s.push_teleports[(nx, ny)]
                    annotations.append({
                        "step": i + 1,
                        "total": len(self.actions),
                        "action": ACTION_NAMES[act],
                        "from": list(pos),
                        "to": list(dest),
                        "phase": "push_block_teleport",
                        "note": f"Push block launches player from {pos} to {dest}",
                    })
                    pos = dest
                    continue
                pos = (nx, ny)

            # Determine what happened at this position
            note = ""
            if pos in coll_positions:
                note = "Collected step refill - budget reset to max"
                coll_positions.discard(pos)
                phase = "collect_refill"
            elif pos in changer_positions:
                for ct, n, cpos in changer_info:
                    if cpos == pos and visits_remaining.get(ct, 0) > 0:
                        visits_remaining[ct] -= 1
                        rem = visits_remaining[ct]
                        note = f"Activated {ct} changer ({n - rem}/{n} visits done)"
                        phase = f"visit_{ct}_changer"
                        break
            elif goal and pos == (goal["x"], goal["y"]):
                note = "Reached goal with correct attributes - level complete!"
                phase = "reach_goal"
            else:
                # Determine current target
                for ct, n, cpos in changer_info:
                    if visits_remaining.get(ct, 0) > 0:
                        current_target = f"{ct} changer at {cpos}"
                        phase = f"navigate_to_{ct}_changer"
                        break
                else:
                    if goal:
                        current_target = f"goal at ({goal['x']},{goal['y']})"
                        phase = "navigate_to_goal"

            annotations.append({
                "step": i + 1,
                "total": len(self.actions),
                "action": ACTION_NAMES[act],
                "from": [pos[0] - dc * self.pw, pos[1] - dr * self.ph],
                "to": list(pos),
                "phase": phase,
                "target": current_target,
                **({"note": note} if note else {}),
            })

        return annotations

    def get_reasoning(self, action_idx):
        """Get the reasoning dict for a specific action index."""
        if action_idx < len(self.annotations):
            return self.annotations[action_idx]
        return {"step": action_idx + 1, "action": "unknown"}

    def get_plan_summary(self):
        """Get a summary reasoning for the RESET/plan phase."""
        s = self.solver
        goal = s.goals[0] if s.goals else None
        summary = {
            "solver": "BFS multi-waypoint with budget-aware collectible routing",
            "level": self.level_idx + 1,
            "player_start": list(s.start_pos),
            "total_planned_actions": len(self.actions),
            "step_budget": s.step_budget,
        }
        if goal:
            summary["goal"] = {
                "position": [goal["x"], goal["y"]],
                "required_shape": goal["shape"],
                "required_color_idx": goal["color_idx"],
                "required_rotation_idx": goal["rotation_idx"],
            }
            summary["current_attrs"] = {
                "shape": s.shape,
                "color_idx": s.color_idx,
                "rotation_idx": s.rotation_idx,
            }
            shape_s = (goal["shape"] - s.shape) % s.n_shapes
            color_s = (goal["color_idx"] - s.color_idx) % s.n_colors
            rot_s = (goal["rotation_idx"] - s.rotation_idx) % 4
            changes = []
            if shape_s > 0:
                changes.append(f"{shape_s}x shape changer")
            if color_s > 0:
                changes.append(f"{color_s}x color changer")
            if rot_s > 0:
                changes.append(f"{rot_s}x rotation changer")
            summary["strategy"] = f"Visit {', '.join(changes) if changes else 'no changers'}, then reach goal"
            summary["collectibles_available"] = len(s.collectibles)
        return summary


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

        # Build reasoning logger for this level's plan
        logger = ReasoningLogger(local_game, solver, actions, level_idx)
        plan_summary = logger.get_plan_summary()
        print(f"  Strategy: {plan_summary.get('strategy', 'N/A')}")

        # Execute via API
        prev_levels = frame.get("levels_completed", 0)
        level_solved = False

        for i, act in enumerate(actions):
            ga = GAME_ACTION_MAP.get(act)
            if ga is None:
                continue

            # Get reasoning for this action
            reasoning_data = logger.get_reasoning(i)
            # Send as raw dict - the API accepts any JSON-serializable value
            try:
                frame = client.action(
                    game_full_id, guid, ga,
                    reasoning=reasoning_data,
                )
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
