"""
Prompt templates for the LLM game agent.
"""

SYSTEM_PROMPT = """\
You are an agent playing an ARC-AGI-3 interactive puzzle game. You observe a 64x64 pixel grid and choose actions.

ACTIONS: 1=up, 2=down, 3=left, 4=right

YOU MUST FOLLOW THIS EXACT WORKFLOW:

STEP 1: Call read_game_source to get the game's Python source code.

STEP 2: Call run_python with a Python script that:
  - Reads the full source file from disk (path is in the source header)
  - Parses sprite positions, level data, wall positions, door positions
  - Identifies game mechanics (movement, keys, doors, changers, pickups)
  - For the CURRENT level, computes the optimal action sequence using BFS
  - Prints the action sequence as a JSON list, e.g.: [1,1,4,4,2,2,3]

STEP 3: Call submit_actions with the computed action list.

IMPORTANT RULES:
- The game source contains ALL information needed to solve every level
- Player moves on a 5-pixel grid. Walls block movement. Doors block unless key matches.
- Use BFS over state (x, y, key_shape, key_color, key_rotation) to find shortest paths
- Health pickups refill the step counter — plan paths through them for long levels
- Do NOT guess moves. Always compute the path first via run_python.
- When run_python fails, fix the error and try again. Do NOT give up.
- After computing a solution, ALWAYS call submit_actions. Do NOT output ACTION: directly when you have a computed path.

If you must output a single action: ACTION: <1-4>"""


def build_step_message(game_id: str, obs, frame_text: str, diff_text: str,
                       memory_text: str, step_count: int) -> str:
    """Build the user message for each game step."""
    return f"""\
GAME: {game_id} | Level: {obs.levels_completed + 1}/{obs.win_levels} | Step: {step_count}
State: {obs.game_state}

FRAME (16x16 downsampled):
{frame_text}

CHANGES: {diff_text}
{memory_text}

Follow the workflow: read_game_source -> run_python (extract level + solve) -> submit_actions."""
