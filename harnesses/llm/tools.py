"""
Tool definitions and executor for the LLM agent.
The LLM can call these tools to analyze the game before deciding actions.
"""

import subprocess
import json
from pathlib import Path
from collections import deque

from ..perception import GameObservation, Frame
from .frame_encoder import encode_rle, encode_downsampled


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_game_source",
            "description": "Read the full Python source code of the current game. Use this to understand game mechanics, level layouts, sprites, win conditions, etc.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a Python script to analyze the game source or compute paths. The script runs in a subprocess with a 30s timeout. You can import json, math, collections, etc. Print results to stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_frame_region",
            "description": "Get exact pixel values for a rectangular region of the current frame. Useful for inspecting specific objects or areas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Left x coordinate (0-63)"},
                    "y": {"type": "integer", "description": "Top y coordinate (0-63)"},
                    "width": {"type": "integer", "description": "Region width"},
                    "height": {"type": "integer", "description": "Region height"},
                },
                "required": ["x", "y", "width", "height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_action_history",
            "description": "Get the last N actions taken and whether they caused frame changes. Helps identify walls, movement patterns, and game mechanics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of recent actions to return (default 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_actions",
            "description": "Submit a sequence of actions to execute. Use this when you have computed a multi-step plan (e.g., a path from source analysis). Actions: 1=up, 2=down, 3=left, 4=right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of action IDs to execute in order",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why these actions",
                    },
                },
                "required": ["actions"],
            },
        },
    },
]


class ToolExecutor:
    """Executes tool calls from the LLM agent."""

    def __init__(self, game_id: str, dataset_dir: str = "dataset"):
        self.game_id = game_id
        self.dataset_dir = Path(dataset_dir)
        self.current_obs: GameObservation | None = None
        self.action_history: deque = deque(maxlen=100)
        self._source_cache: str | None = None
        self._pending_actions: list[int] = []

    def update(self, obs: GameObservation, action_id: int | None = None,
               frame_changed: bool = False):
        """Update state after a game step."""
        self.current_obs = obs
        if action_id is not None:
            names = {1: "up", 2: "down", 3: "left", 4: "right", 5: "action5", 6: "click", 7: "undo"}
            self.action_history.append({
                "action": action_id,
                "name": names.get(action_id, f"action{action_id}"),
                "frame_changed": frame_changed,
                "levels_completed": obs.levels_completed,
                "game_state": obs.game_state,
            })

    def has_pending_actions(self) -> bool:
        return len(self._pending_actions) > 0

    def pop_action(self) -> int | None:
        if self._pending_actions:
            return self._pending_actions.pop(0)
        return None

    def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "read_game_source":
                return self._read_source()
            elif tool_name == "run_python":
                return self._run_python(args.get("code", ""))
            elif tool_name == "get_frame_region":
                return self._get_frame_region(
                    args.get("x", 0), args.get("y", 0),
                    args.get("width", 16), args.get("height", 16))
            elif tool_name == "get_action_history":
                return self._get_history(args.get("count", 20))
            elif tool_name == "submit_actions":
                return self._submit_actions(
                    args.get("actions", []), args.get("reasoning", ""))
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {e}"

    def _read_source(self) -> str:
        if self._source_cache:
            return self._source_cache
        source_path = self.dataset_dir / "games" / self.game_id / "source.py"
        if not source_path.exists():
            return f"Source not found at {source_path}"
        text = source_path.read_text()
        # Truncate if extremely long (>30k chars)
        if len(text) > 30000:
            text = text[:30000] + "\n... [truncated]"
        self._source_cache = text
        return text

    def _run_python(self, code: str) -> str:
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.dataset_dir.parent),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\nSTDERR: {result.stderr[:1000]}"
            return output[:5000]  # cap output
        except subprocess.TimeoutExpired:
            return "ERROR: Script timed out (30s limit)"
        except Exception as e:
            return f"ERROR: {e}"

    def _get_frame_region(self, x: int, y: int, w: int, h: int) -> str:
        if self.current_obs is None:
            return "No frame available"
        arr = self.current_obs.frame.to_numpy()
        x = max(0, min(x, 63))
        y = max(0, min(y, 63))
        w = min(w, 64 - x)
        h = min(h, 64 - y)
        region = arr[y:y+h, x:x+w]
        lines = []
        for row_idx, row in enumerate(region):
            lines.append(f"{y+row_idx:2d}|" + " ".join(f"{v:2d}" for v in row))
        return "\n".join(lines)

    def _get_history(self, count: int = 20) -> str:
        recent = list(self.action_history)[-count:]
        if not recent:
            return "No actions taken yet"
        lines = []
        for i, h in enumerate(recent):
            changed = "moved" if h["frame_changed"] else "BLOCKED"
            lines.append(f"  {i}: {h['name']} -> {changed} (level {h['levels_completed']})")
        return "\n".join(lines)

    def _submit_actions(self, actions: list, reasoning: str) -> str:
        if not actions:
            return "No actions provided"
        valid = [a for a in actions if a in (1, 2, 3, 4, 5, 6, 7)]
        self._pending_actions = valid
        return f"Queued {len(valid)} actions. They will execute on subsequent steps."
