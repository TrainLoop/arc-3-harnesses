"""
Tool definitions and executor for the LLM agent.
The LLM can call these tools to analyze the game before deciding actions.
Supports all action types: directional (1-5), click (6 with x,y), undo (7).
"""

import subprocess
import sys
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
            "description": "Read the Python source code of the current game. Contains ALL game mechanics, sprites, levels, win conditions. Call this FIRST for any new game.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a Python script to analyze game source or compute solutions. Has access to arcengine, numpy, json, math, collections. The game source file path is shown in the read_game_source output. Print results to stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Has 30s timeout.",
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
            "description": "Get exact pixel color values for a rectangular region of the current 64x64 frame. Useful for inspecting specific sprites or areas.",
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
            "description": "Get the last N actions taken and whether they caused frame changes. Helps identify walls, patterns, and game responses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of recent actions (default 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_actions",
            "description": "Submit a sequence of actions to execute. For directional games: [1,2,3,4] = up/down/left/right. For click games: provide actions list AND clicks list with [x,y] coordinates for each action 6.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of action IDs to execute in order (1-7)",
                    },
                    "clicks": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "description": "For action 6 (click): list of [x, y] coordinates. One per action-6 in the actions list.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the plan",
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
        self._pending_clicks: list[list[int]] = []  # for action 6
        self._click_index: int = 0

    def update(self, obs: GameObservation, action_id: int | None = None,
               frame_changed: bool = False):
        self.current_obs = obs
        if action_id is not None:
            names = {1: "up", 2: "down", 3: "left", 4: "right",
                     5: "action5", 6: "click", 7: "undo"}
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

    def pop_click(self) -> list[int] | None:
        """Pop the next click coordinate for action 6."""
        if self._pending_clicks:
            return self._pending_clicks.pop(0)
        return None

    def execute(self, tool_name: str, args: dict) -> str:
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
                    args.get("actions", []),
                    args.get("clicks", []),
                    args.get("reasoning", ""))
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {e}"

    def _read_source(self) -> str:
        if self._source_cache:
            return self._source_cache
        # Try short game id first, then with version
        source_path = self.dataset_dir / "games" / self.game_id / "source.py"
        if not source_path.exists():
            short_id = self.game_id.split("-")[0] if "-" in self.game_id else self.game_id
            source_path = self.dataset_dir / "games" / short_id / "source.py"
        if not source_path.exists():
            return f"Source not found for {self.game_id}"
        text = source_path.read_text()

        # Pre-extract structured game info by actually loading the game
        struct_info = self._extract_game_info(source_path)
        # For large sources, only include the game class (not sprites/levels raw text)
        summary = self._summarize_source(text, source_path)
        if struct_info:
            # Structured info is much more useful than raw source — put it first
            result = struct_info + "\n\n" + summary
        else:
            result = summary
        self._source_cache = result
        return result

    def _extract_game_info(self, source_path: Path) -> str:
        """Load the game and extract structured info for the LLM."""
        python = str(Path(sys.executable))
        code = f'''
import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("gm", "{source_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from arcengine import ARCBaseGame, GameAction, ActionInput
game = None
for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
        game = obj()
        break
if not game:
    print("ERROR: no game class found")
    sys.exit(1)
game.perform_action(ActionInput(id=GameAction.RESET))
info = {{}}
info["class_name"] = type(game).__name__
info["num_levels"] = len(game._levels)
info["available_actions"] = game._available_actions if hasattr(game, "_available_actions") else []
info["grid_size"] = list(game.current_level.grid_size) if game.current_level.grid_size else [64,64]
# Extract per-level data dicts
info["levels"] = []
for i, lvl in enumerate(game._clean_levels):
    d = {{}}
    d["index"] = i
    if hasattr(lvl, "_data") and lvl._data:
        d["data"] = dict(lvl._data)
    elif hasattr(lvl, "data") and callable(getattr(lvl, "get_data", None)):
        pass  # data access requires level to be set
    # Count sprites by tag
    tags = {{}}
    for s in lvl._sprites:
        if s.tags:
            for t in s.tags:
                tags[t] = tags.get(t, 0) + 1
    d["sprite_tags"] = tags
    d["num_sprites"] = len(lvl._sprites)
    info["levels"].append(d)
# Current level state
game.set_level(0)
cl = game.current_level
info["current_level_data"] = {{}}
if hasattr(cl, "_data") and cl._data:
    info["current_level_data"] = {{k: v for k, v in cl._data.items()}}
# Sprite positions for current level
sprite_info = []
for s in cl._sprites:
    si = {{"name": s.name, "x": s.x, "y": s.y, "visible": s.is_visible}}
    if s.tags:
        si["tags"] = list(s.tags)
    si["w"] = s.width
    si["h"] = s.height
    sprite_info.append(si)
info["level1_sprites"] = sprite_info
print(json.dumps(info, indent=2, default=str))
'''
        try:
            result = subprocess.run(
                [python, "-c", code],
                capture_output=True, text=True, timeout=15,
                cwd=str(self.dataset_dir.parent),
            )
            if result.returncode == 0 and result.stdout.strip():
                return "# === PRE-EXTRACTED GAME INFO (JSON) ===\n" + result.stdout.strip()
            return ""
        except Exception:
            return ""

    def _summarize_source(self, text: str, path) -> str:
        """Extract the most useful parts of a game source for the LLM.
        Provides pre-parsed structure so the model doesn't have to regex raw source."""
        import re as _re
        lines = text.split('\n')
        parts = [f"# Game source: {path} ({len(lines)} lines total)"]
        parts.append(f"# To read full source in run_python: open('{path}').read()")

        # Extract class name
        class_name = None
        class_start = None
        for i, line in enumerate(lines):
            m = _re.match(r'class (\w+)\(ARCBaseGame\)', line)
            if m:
                class_name = m.group(1)
                class_start = i
                break
        if class_name:
            parts.append(f"\n# Game class: {class_name}(ARCBaseGame)")

        # Extract available_actions from __init__
        for line in lines:
            if 'available_actions' in line and '[' in line:
                parts.append(f"# {line.strip()}")
                break

        # Extract constants (color values, timing, etc)
        parts.append("\n# === CONSTANTS ===")
        for line in lines:
            stripped = line.strip()
            if ('=' in stripped and not stripped.startswith(' ') and
                not stripped.startswith('#') and not stripped.startswith('sprites') and
                not stripped.startswith('levels') and not stripped.startswith('"') and
                not stripped.startswith('class ') and not stripped.startswith('def ')):
                if any(c.isdigit() for c in stripped) and len(stripped) < 80:
                    parts.append(stripped)

        # Extract level data dicts (most important for solving)
        parts.append("\n# === LEVEL DATA (extracted data= dicts) ===")
        import ast
        level_num = 0
        for i, line in enumerate(lines):
            if '# Level' in line:
                level_num += 1
                parts.append(f"\n{line.strip()}")
            if 'data={' in line or 'data =' in line:
                # Grab the data dict block
                block = []
                for j in range(i, min(i + 25, len(lines))):
                    block.append(lines[j])
                    if '}' in lines[j] and lines[j].strip().endswith('},') or lines[j].strip().endswith('},'):
                        break
                    if lines[j].strip() == '},':
                        break
                parts.append('    ' + '\n    '.join(l.strip() for l in block))

        # Include full game class (the critical logic)
        if class_start is not None:
            game_logic = '\n'.join(lines[class_start:])
            parts.append("\n# === GAME CLASS (full logic) ===\n")
            parts.append(game_logic)

        result = '\n'.join(parts)
        if len(result) > 15000:
            result = result[:15000] + f"\n\n# ... [truncated — read full source: open('{path}').read()]"
        return result

    def _run_python(self, code: str) -> str:
        # Use the venv python so arcengine and numpy are available
        python = str(Path(sys.executable))
        try:
            result = subprocess.run(
                [python, "-c", code],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.dataset_dir.parent),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\nSTDERR: {result.stderr[:2000]}"
            return output[:8000]
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

    def _submit_actions(self, actions: list, clicks: list = None, reasoning: str = "") -> str:
        if not actions:
            return "No actions provided"
        valid = [a for a in actions if a in (1, 2, 3, 4, 5, 6, 7)]
        self._pending_actions = valid
        self._pending_clicks = clicks or []
        self._click_index = 0
        note = f" (reasoning: {reasoning})" if reasoning else ""
        click_note = f", {len(self._pending_clicks)} click coords" if self._pending_clicks else ""
        return f"Queued {len(valid)} actions{click_note}{note}. They will execute on subsequent steps."
