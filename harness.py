#!/usr/bin/env python3
"""
Claude Code Harness for ARC-AGI-3 Games

Uses Claude Code CLI to analyze game source code and solve each level.
Uses arcengine to load games locally and get per-level frames.

Usage:
    python harness.py --game ls20
    python harness.py --game ls20 --model opus
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "ACTION5", 6: "ACTION6", 7: "ACTION7"}
NAME_TO_ACTION = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4,
                  "U": 1, "D": 2, "L": 3, "R": 4}

# All valid GameAction IDs (some games use 5/6/7 for cycle/click/undo)
VALID_ACTIONS = {1, 2, 3, 4, 5, 6, 7}


# ============================================================
# Game Loading (via arcengine)
# ============================================================

def load_game(game_dir: str):
    """Load game class from source.py and instantiate it."""
    source_path = os.path.join(game_dir, "source.py")
    spec = importlib.util.spec_from_file_location("game_mod", source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from arcengine import ARCBaseGame
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
            return obj()
    raise ValueError(f"No ARCBaseGame subclass found in {source_path}")


def render_frame_text(frame) -> str:
    """Render a 64x64 frame as a compact text grid for Claude Code."""
    if isinstance(frame[0], list) and isinstance(frame[0][0], list):
        frame = frame[-1]
    h, w = len(frame), len(frame[0])
    step = 5
    lines = []
    header = "    " + "".join(f"{x // step:3d}" for x in range(0, w, step))
    lines.append(header)
    for y in range(0, h, step):
        row_label = f"{y // step:3d} "
        cells = []
        for x in range(0, w, step):
            cells.append(f"{frame[y][x]:3d}")
        lines.append(row_label + "".join(cells))
    return "\n".join(lines)


# ============================================================
# Claude Code CLI Interface
# ============================================================

def call_claude_code(prompt: str, model: str = None, timeout: int = 1200) -> tuple:
    """Call Claude Code CLI with stream-json and return (response_text, metadata, trace).

    Returns:
        response: Final text response from Claude.
        metadata: Dict with token usage, cost, timing, turn info.
        trace: List of trace events, each a dict with keys:
            - turn: int (which assistant turn, 1-indexed)
            - type: "thinking" | "tool_call" | "tool_result" | "text"
            - content: str (thinking text, tool input, tool output, or response text)
            - tool_name: str (only for tool_call/tool_result)
            - tokens: dict (per-turn token usage from the assistant message, if available)
    """
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
           "--dangerously-skip-permissions"]

    if model:
        cmd.extend(["--model", model])

    empty_meta = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "num_turns": 0, "total_cost_usd": 0, "duration_ms": 0,
        "context_window": 0, "max_output_tokens": 0,
    }

    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Claude Code timed out after {timeout}s, killing process group")
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()
        return "", {**empty_meta, "timed_out": True}, []

    stderr_text = (stderr or "").strip()
    stdout_text = (stdout or "").strip()

    if proc.returncode != 0:
        print(f"  [ERROR] Claude Code exited with code {proc.returncode}")
        print(f"  [ERROR] stderr: {stderr_text[:1000]}")
        return "", empty_meta, []

    if not stdout_text:
        print(f"  [ERROR] Claude Code returned empty stdout")
        return "", empty_meta, []

    # Parse stream-json: one JSON object per line
    events = []
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Build trace and extract result
    trace = []
    response = ""
    meta = dict(empty_meta)
    turn_num = 0
    # Map tool_use_id -> tool_name for correlating tool results
    tool_id_to_name = {}

    for evt in events:
        evt_type = evt.get("type", "")

        if evt_type == "assistant":
            turn_num += 1
            msg = evt.get("message", {})
            usage = msg.get("usage", {})
            turn_tokens = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            }
            for block in msg.get("content", []):
                block_type = block.get("type", "")
                if block_type == "thinking":
                    trace.append({
                        "turn": turn_num, "type": "thinking",
                        "content": block.get("thinking", ""),
                        "tokens": turn_tokens,
                    })
                elif block_type == "tool_use":
                    tool_name = block.get("name", "")
                    tool_id = block.get("id", "")
                    tool_id_to_name[tool_id] = tool_name
                    trace.append({
                        "turn": turn_num, "type": "tool_call",
                        "tool_name": tool_name,
                        "content": json.dumps(block.get("input", {}), indent=2),
                        "tokens": turn_tokens,
                    })
                elif block_type == "text":
                    trace.append({
                        "turn": turn_num, "type": "text",
                        "content": block.get("text", ""),
                        "tokens": turn_tokens,
                    })

        elif evt_type == "user":
            # Tool results come inside "user" events
            msg = evt.get("message", {})
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    tool_name = tool_id_to_name.get(tool_id, "unknown")
                    content = block.get("content", "")
                    if isinstance(content, list):
                        parts = []
                        for c in content:
                            if isinstance(c, dict):
                                parts.append(c.get("text", str(c)))
                            else:
                                parts.append(str(c))
                        content = "\n".join(parts)
                    trace.append({
                        "turn": turn_num, "type": "tool_result",
                        "tool_name": tool_name,
                        "content": str(content),
                    })

        elif evt_type == "result":
            response = evt.get("result", "")
            usage = evt.get("usage", {})
            model_usage = evt.get("modelUsage", {})
            model_info = next(iter(model_usage.values()), {}) if model_usage else {}

            meta = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "num_turns": evt.get("num_turns", 0),
                "total_cost_usd": evt.get("total_cost_usd", 0),
                "duration_ms": evt.get("duration_ms", 0),
                "duration_api_ms": evt.get("duration_api_ms", 0),
                "context_window": model_info.get("contextWindow", 0),
                "max_output_tokens": model_info.get("maxOutputTokens", 0),
                "stop_reason": evt.get("stop_reason", ""),
                "session_id": evt.get("session_id", ""),
            }

    total_input = meta["input_tokens"] + meta["cache_creation_input_tokens"] + meta["cache_read_input_tokens"]
    meta["total_tokens"] = total_input + meta["output_tokens"]
    if meta["context_window"] > 0:
        meta["context_utilization_pct"] = round(100.0 * total_input / meta["context_window"], 1)
    else:
        meta["context_utilization_pct"] = 0

    # Summarize trace
    n_thinking = sum(1 for t in trace if t["type"] == "thinking")
    n_tool_calls = sum(1 for t in trace if t["type"] == "tool_call")
    thinking_chars = sum(len(t["content"]) for t in trace if t["type"] == "thinking")

    print(f"  [USAGE] input={meta['input_tokens']:,} output={meta['output_tokens']:,} "
          f"cache_create={meta['cache_creation_input_tokens']:,} cache_read={meta['cache_read_input_tokens']:,}")
    print(f"  [USAGE] total_tokens={meta['total_tokens']:,} turns={meta['num_turns']} "
          f"cost=${meta['total_cost_usd']:.2f} context_used={meta['context_utilization_pct']}%")
    print(f"  [TRACE] {n_thinking} thinking blocks ({thinking_chars:,} chars), "
          f"{n_tool_calls} tool calls, {turn_num} assistant turns")

    if response:
        print(f"  [OK] Got response ({len(response)} chars)")
    else:
        print(f"  [WARN] No result text in response")

    return response, meta, trace


def _is_valid_action(a) -> bool:
    """Check if an action entry is valid: int 1-7 or [6, x, y] click."""
    if isinstance(a, int) and a in VALID_ACTIONS:
        return True
    if isinstance(a, list) and len(a) == 3 and a[0] == 6:
        return all(isinstance(v, int) for v in a)
    return False


def _action_label(a) -> str:
    """Human-readable label for an action entry."""
    if isinstance(a, list) and len(a) == 3 and a[0] == 6:
        return f"CLICK({a[1]},{a[2]})"
    return ACTION_NAMES.get(a, f"?{a}")


def parse_move_sequence(text: str) -> list:
    """Extract a move sequence from text.

    Accepts action IDs 1-7 and click tuples [6, x, y].
    Prefers arrays inside ```json fenced blocks, then bare JSON arrays,
    then direction names.
    """
    # First: look for ```json [...] ``` fenced blocks (may contain nested [6,x,y])
    for match in re.finditer(r'```json\s*\n?\s*(\[.+?\])\s*\n?\s*```', text, re.DOTALL):
        try:
            actions = json.loads(match.group(1))
            if (isinstance(actions, list) and len(actions) >= 2
                    and all(_is_valid_action(a) for a in actions)):
                return actions
        except (json.JSONDecodeError, TypeError):
            continue

    # Then: bare JSON arrays, require length >= 5 to avoid coordinate/table lists
    for match in re.finditer(r'\[[\d\s,\[\]]+\]', text):
        try:
            actions = json.loads(match.group())
            if (isinstance(actions, list) and len(actions) >= 5
                    and all(_is_valid_action(a) for a in actions)):
                return actions
        except (json.JSONDecodeError, TypeError):
            continue

    # Try comma/space-separated direction names (only UPPERCASE to avoid prose matches)
    name_matches = re.findall(r'\b(UP|DOWN|LEFT|RIGHT)\b', text)
    if len(name_matches) >= 5:
        return [NAME_TO_ACTION[n] for n in name_matches]

    return []


def parse_all_level_moves(analysis_text: str, n_levels: int) -> dict:
    """Parse per-level move sequences from the analysis response.

    Looks for sections like "### Level N" followed by a JSON array of
    action IDs (preferably in a ```json block).
    Returns {level_num: [actions]} for each level found.
    """
    result = {}

    # Split by level headers
    level_pattern = re.compile(
        r'(?:^|\n)(?:#{1,4}\s*)?Level\s+(\d+)\b', re.IGNORECASE
    )
    matches = list(level_pattern.finditer(analysis_text))

    for i, m in enumerate(matches):
        level_num = int(m.group(1))
        if level_num < 1 or level_num > n_levels:
            continue

        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(analysis_text)
        section = analysis_text[start:end]

        moves = parse_move_sequence(section)
        if moves:
            result[level_num] = moves

    return result


# ============================================================
# Main Harness
# ============================================================

def run_game(game_id: str, base_dir: Path, model: str = None,
             run_dir: str = None):
    """Run the full harness for a single game."""
    from arcengine import ActionInput, GameAction

    game_dir = base_dir / "dataset" / "games" / game_id
    source_path = game_dir / "source.py"
    meta_path = game_dir / "metadata.json"

    if not source_path.exists():
        print(f"Error: {source_path} not found")
        sys.exit(1)

    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    game_full_id = metadata.get("game_id", game_id)

    # Load game locally via arcengine
    game = load_game(str(game_dir))
    reset_frame = game.perform_action(ActionInput(id=GameAction.RESET))
    win_levels = reset_frame.win_levels
    available_actions = reset_frame.available_actions

    print(f"Game: {game_full_id}")
    print(f"Levels: {win_levels}")
    print(f"Available actions: {available_actions}")

    # Prepare output directory: output/runs/<run_dir>/<game_id>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir:
        out_dir = base_dir / "output" / "runs" / run_dir / game_id
    else:
        out_dir = base_dir / "output" / game_id
    os.makedirs(out_dir, exist_ok=True)
    reasoning_path = out_dir / f"{game_id}_reasoning.txt"
    moves_path = out_dir / f"{game_id}_moves.txt"

    reasoning_lines: List[str] = []
    moves_lines: List[str] = []

    # Build action name string from available actions
    action_desc_parts = []
    for a in sorted(available_actions):
        name = ACTION_NAMES.get(a, f"ACTION{a}")
        action_desc_parts.append(f"{a}={name}")
    action_desc = ", ".join(action_desc_parts)

    # ------------------------------------------------------------------
    # Single-pass: ask Claude Code to analyze AND produce moves
    # ------------------------------------------------------------------
    print(f"\n--- Analyzing source code and solving all levels ---")
    # Build click instruction if ACTION6 is available
    click_instruction = ""
    if 6 in available_actions:
        click_instruction = (
            "\n\nIMPORTANT: For click actions (ACTION6), encode as [6, x, y] where x,y "
            "are the pixel coordinates to click. Example with clicks:\n"
            "```json\n[4, 2, [6, 45, 33], 3, 3, [6, 12, 50], 5]\n```\n"
            "Each [6, x, y] entry clicks at screen position (x, y). "
            "Read the source code to determine the correct coordinates for buttons, "
            "gates, tiles, or other clickable elements."
        )

    analysis_prompt = (
        f"Read {source_path}. This is an ARC-AGI-3 game with {win_levels} levels.\n\n"
        f"Available actions: {action_desc}\n\n"
        f"For EACH level (1 through {win_levels}), analyze the game mechanics and "
        f"produce the exact move sequence to beat it.\n\n"
        f"For each level, end your analysis with the moves as a JSON array, e.g.:\n"
        f"```json\n[3, 3, 1, 1, 4, 4, 2]\n```\n"
        f"{click_instruction}\n\n"
        f"Make sure every level has a ```json [...] ``` block with the action sequence."
    )
    t0 = time.time()
    # 3 hour timeout
    timeout = 10800
    print(f"  Timeout: {timeout}s")
    analysis_resp, analysis_meta, analysis_trace = call_claude_code(
        analysis_prompt, model=model, timeout=timeout)
    analysis_elapsed = time.time() - t0

    reasoning_lines.append("=" * 60)
    reasoning_lines.append("SOURCE CODE ANALYSIS & SOLUTIONS")
    reasoning_lines.append("=" * 60)
    reasoning_lines.append(analysis_resp or "(no response)")
    reasoning_lines.append("")

    # Save analysis
    analysis_file = out_dir / f"{game_id}_analysis.txt"
    with open(analysis_file, "w") as f:
        f.write(analysis_resp or "(no response)")

    # Save reasoning trace
    trace_file = out_dir / f"{game_id}_trace.txt"
    with open(trace_file, "w") as f:
        for entry in analysis_trace:
            turn = entry.get("turn", "?")
            etype = entry["type"]
            tool = entry.get("tool_name", "")
            content = entry["content"]
            tokens = entry.get("tokens", {})

            f.write(f"{'='*60}\n")
            header = f"[Turn {turn}] {etype.upper()}"
            if tool:
                header += f" ({tool})"
            if tokens:
                tok_in = tokens.get("input_tokens", 0)
                tok_out = tokens.get("output_tokens", 0)
                header += f"  [in={tok_in:,} out={tok_out:,}]"
            f.write(header + "\n")
            f.write(f"{'='*60}\n")
            f.write(content + "\n\n")

    print(f"  Analysis complete ({analysis_elapsed:.1f}s). Saved to {analysis_file}")
    print(f"  Trace: {trace_file} ({len(analysis_trace)} events)")

    # Parse moves for all levels from the analysis
    all_level_moves = parse_all_level_moves(analysis_resp or "", win_levels)
    print(f"  Parsed moves for levels: {sorted(all_level_moves.keys())}")

    # ------------------------------------------------------------------
    # Execute moves for each level
    # ------------------------------------------------------------------
    game_action_map = {
        1: GameAction.ACTION1, 2: GameAction.ACTION2,
        3: GameAction.ACTION3, 4: GameAction.ACTION4,
    }
    # Add optional actions if available
    for act_id, ga_name in [(5, "ACTION5"), (6, "ACTION6"), (7, "ACTION7")]:
        if hasattr(GameAction, ga_name):
            game_action_map[act_id] = getattr(GameAction, ga_name)

    levels_solved = 0
    latest_frame = reset_frame
    level_stats: List[dict] = []

    for level_idx in range(win_levels):
        level_num = level_idx + 1
        print(f"\n{'=' * 60}")
        print(f"LEVEL {level_num}/{win_levels}")

        actions = all_level_moves.get(level_num, [])
        if not actions:
            print(f"  No moves found in analysis for level {level_num}")
            moves_lines.append(f"Level {level_num}: FAILED (no moves in analysis)")
            level_stats.append({
                "level": level_num, "solved": False, "moves": 0,
                "time_seconds": 0, "reason": "no moves parsed",
            })
            break

        move_names = [_action_label(a) for a in actions]
        print(f"  Moves ({len(actions)}): {', '.join(move_names[:30])}{'...' if len(actions) > 30 else ''}")
        moves_lines.append(f"Level {level_num}: {', '.join(move_names)}")

        # Execute the moves on the local game engine
        prev_level_idx = game._current_level_index
        level_solved = False
        fail_reason = "moves exhausted"

        for i, act in enumerate(actions):
            # Handle click actions: [6, x, y]
            if isinstance(act, list) and len(act) == 3 and act[0] == 6:
                ga = game_action_map.get(6)
                if ga is None:
                    continue
                frame_data = game.perform_action(
                    ActionInput(id=ga, data={"x": act[1], "y": act[2]}))
            else:
                ga = game_action_map.get(act)
                if ga is None:
                    continue
                frame_data = game.perform_action(ActionInput(id=ga))

            if game._current_level_index != prev_level_idx:
                level_solved = True
                latest_frame = frame_data
                print(f"  Level {level_num} SOLVED after {i + 1} actions!")
                break

            state_str = str(frame_data.state)
            if "WIN" in state_str:
                level_solved = True
                latest_frame = frame_data
                print(f"  WIN after {i + 1} actions!")
                break
            if "GAME_OVER" in state_str:
                fail_reason = "game over"
                print(f"  GAME OVER at action {i + 1}")
                break

        level_stats.append({
            "level": level_num,
            "solved": level_solved,
            "moves": len(actions) if level_solved else 0,
            "time_seconds": 0,  # all reasoning happened in the analysis pass
            "reason": "solved" if level_solved else fail_reason,
        })

        if level_solved:
            levels_solved += 1
        else:
            print(f"  Level {level_num} FAILED")
            break

        if "WIN" in str(latest_frame.state):
            break

    # ------------------------------------------------------------------
    # Save output files
    # ------------------------------------------------------------------
    with open(reasoning_path, "w") as f:
        f.write("\n".join(reasoning_lines))

    with open(moves_path, "w") as f:
        f.write("\n".join(moves_lines))

    # Build trace summary
    trace_summary = {
        "total_events": len(analysis_trace),
        "thinking_blocks": sum(1 for t in analysis_trace if t["type"] == "thinking"),
        "thinking_chars": sum(len(t["content"]) for t in analysis_trace if t["type"] == "thinking"),
        "tool_calls": sum(1 for t in analysis_trace if t["type"] == "tool_call"),
        "tool_results": sum(1 for t in analysis_trace if t["type"] == "tool_result"),
        "text_blocks": sum(1 for t in analysis_trace if t["type"] == "text"),
        "assistant_turns": max((t.get("turn", 0) for t in analysis_trace), default=0),
        "tools_used": {},
    }
    for t in analysis_trace:
        if t["type"] == "tool_call":
            name = t.get("tool_name", "unknown")
            trace_summary["tools_used"][name] = trace_summary["tools_used"].get(name, 0) + 1

    stats = {
        "game_id": game_full_id,
        "timestamp": timestamp,
        "levels_solved": levels_solved,
        "total_levels": win_levels,
        "analysis_time_seconds": round(analysis_elapsed, 1),
        "level_stats": level_stats,
        "context_usage": analysis_meta,
        "trace_summary": trace_summary,
    }
    stats_path = out_dir / f"{game_id}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Generate plots
    plots_dir = out_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    generate_plots(stats, plots_dir, game_id)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {levels_solved}/{win_levels} levels solved")
    print(f"Reasoning log: {reasoning_path}")
    print(f"Moves log:     {moves_path}")
    print(f"Stats:         {stats_path}")
    print(f"Plots:         {plots_dir}/")

    return levels_solved, win_levels


def generate_plots(stats: dict, plots_dir: Path, game_id: str):
    """Generate plots for time, score, and context usage."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots")
        return

    level_stats = stats["level_stats"]
    total_levels = stats["total_levels"]
    analysis_time = stats["analysis_time_seconds"]
    ctx = stats.get("context_usage", {})

    levels = [s["level"] for s in level_stats]
    solved = [s["solved"] for s in level_stats]

    # Time plot: step function over wall-clock (all reasoning in analysis)
    timestamps_plot = [0, analysis_time]
    levels_won = [0, 0]
    for s in level_stats:
        if s["solved"]:
            timestamps_plot.append(analysis_time)
            levels_won.append(levels_won[-1])
            timestamps_plot.append(analysis_time)
            levels_won.append(levels_won[-1] + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timestamps_plot, levels_won, color="#1565C0", linewidth=2)
    ax.axvline(x=analysis_time, color="#2E7D32", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Levels won")
    ax.set_title(f"{game_id}")
    ax.set_yticks(range(0, total_levels + 1))
    ax.set_ylim(-0.2, total_levels + 0.2)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{game_id}_time.png", dpi=150)
    plt.close(fig)

    # Score plot: step function over cumulative steps
    cum_steps = [0]
    cum_solved = [0]
    running_steps = 0
    running_solved = 0
    for s in level_stats:
        if s["solved"]:
            running_steps += s["moves"]
            cum_steps.append(running_steps)
            cum_solved.append(running_solved)
            running_solved += 1
            cum_steps.append(running_steps)
            cum_solved.append(running_solved)

    if cum_steps[-1] > 0:
        cum_steps.append(cum_steps[-1])
        cum_solved.append(cum_solved[-1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cum_steps, cum_solved, color="#1565C0", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Levels won")
    ax.set_title(f"{game_id} ({running_solved}/{total_levels})")
    ax.set_yticks(range(0, total_levels + 1))
    ax.set_ylim(-0.2, total_levels + 0.2)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{game_id}_score.png", dpi=150)
    plt.close(fig)

    # Context usage plot (if metadata available)
    if ctx and ctx.get("context_window", 0) > 0:
        context_window = ctx["context_window"]
        input_tok = ctx.get("input_tokens", 0)
        cache_create = ctx.get("cache_creation_input_tokens", 0)
        cache_read = ctx.get("cache_read_input_tokens", 0)
        output_tok = ctx.get("output_tokens", 0)
        total_input = input_tok + cache_create + cache_read
        unused = max(0, context_window - total_input)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: token breakdown stacked bar
        ax = axes[0]
        categories = ["Input\n(new)", "Cache\nCreate", "Cache\nRead", "Output"]
        values = [input_tok, cache_create, cache_read, output_tok]
        colors = ["#1565C0", "#2E7D32", "#7CB342", "#F57C00"]
        ax.bar(categories, values, color=colors)
        ax.set_ylabel("Tokens")
        ax.set_title(f"{game_id} — Token Breakdown")
        for i, v in enumerate(values):
            if v > 0:
                ax.text(i, v + context_window * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=8)

        # Right: context utilization pie
        ax = axes[1]
        sizes = [input_tok, cache_create, cache_read, unused]
        labels = [f"Input ({input_tok:,})", f"Cache Create ({cache_create:,})",
                  f"Cache Read ({cache_read:,})", f"Unused ({unused:,})"]
        pie_colors = ["#1565C0", "#2E7D32", "#7CB342", "#E0E0E0"]
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 8})
        ax.set_title(f"Context Utilization: {ctx.get('context_utilization_pct', 0)}% of {context_window:,}")

        fig.tight_layout()
        fig.savefig(plots_dir / f"{game_id}_context.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Claude Code Harness for ARC-AGI-3")
    parser.add_argument("--game", required=True,
                        help="Game folder ID (one of the 25 folder names in dataset/games/)")
    parser.add_argument("--model", default=None,
                        help="Claude model to use (e.g., opus, sonnet)")
    parser.add_argument("--run-dir", default=None,
                        help="Shared run directory name (e.g., run_20260328_010000). "
                             "Outputs go to output/runs/<run-dir>/<game>/")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    games_dir = base_dir / "dataset" / "games"
    valid_ids = sorted(d.name for d in games_dir.iterdir()
                       if d.is_dir() and (d / "source.py").exists())
    if args.game not in valid_ids:
        print(f"Error: Unknown game '{args.game}'")
        print(f"Valid games: {', '.join(valid_ids)}")
        sys.exit(1)

    print(f"ARC-AGI-3 Claude Code Harness")
    print(f"{'=' * 60}")

    run_game(args.game, base_dir, model=args.model, run_dir=args.run_dir)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
