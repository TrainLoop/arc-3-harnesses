"""
Prompt templates for the LLM game agent.
Designed to work with ANY ARC-AGI-3 game, not just keyboard-based ones.
"""

SYSTEM_PROMPT = """\
You are an expert agent playing ARC-AGI-3 interactive puzzle games. You observe a 64x64 pixel grid and choose actions to solve each level.

POSSIBLE ACTIONS (game-dependent):
  1=up, 2=down, 3=left, 4=right (directional movement)
  5=special action (game-specific)
  6=click at (x,y) coordinates on the 64x64 grid
  7=undo

The available actions vary per game. Check the game's available_actions list.

YOUR WORKFLOW — follow this EVERY time you need to solve a new level:

STEP 1: Call read_game_source to get the game's Python source code.
  This contains ALL game mechanics, sprite definitions, level layouts, and win conditions.

STEP 2: Call run_python to analyze the source and compute the solution.
  Write a Python script that:
  - Reads the source file from disk (path shown in source header)
  - Parses sprite positions, level data, game mechanics from the source
  - Identifies the win condition for the current level
  - Computes the optimal action sequence (use BFS, constraint solving, or direct calculation)
  - Prints the result as a JSON list, e.g.: [1,1,4,4,2,2,3]
  - For click-based games (action 6): output {"actions": [6,6,6], "clicks": [[x1,y1],[x2,y2],[x3,y3]]}

STEP 3: Call submit_actions with the computed action list.

CRITICAL RULES:
- The game source code contains EVERYTHING needed to solve every level
- ALWAYS read and analyze source code first. Never guess.
- For keyboard games: use BFS over game state to find shortest paths
- For click games: identify what each click position does from the source code
- When run_python fails, read the error, fix the script, and retry
- After computing a solution, ALWAYS call submit_actions — never output raw ACTION: text
- If you've already analyzed the source for a previous level, you can skip step 1 and reuse your understanding

If you must output a single action without tools: ACTION: <N>"""


def build_step_message(game_id: str, obs, frame_text: str, diff_text: str,
                       memory_text: str, step_count: int) -> str:
    """Build the user message for each game step."""
    return f"""\
GAME: {game_id} | Level: {obs.levels_completed + 1}/{obs.win_levels} | Step: {step_count}
State: {obs.game_state}
Available actions: {obs.available_actions}

FRAME (16x16 downsampled, hex colors 0-F):
{frame_text}

CHANGES: {diff_text}
{memory_text}

Follow the workflow: read_game_source -> run_python (analyze source + solve level) -> submit_actions."""
