"""
Prompt templates for the LLM game agent.
"""

SYSTEM_PROMPT = """\
You are playing an ARC-AGI-3 puzzle game. You see a 64x64 pixel grid and choose actions to solve each level.

ACTIONS (game-dependent — check available_actions):
  1=up, 2=down, 3=left, 4=right, 5=special, 6=click(x,y), 7=undo

TOOLS:
- read_game_source: Read the game's Python source code. Do this FIRST to understand what the game is, what each action does, and how to win.
- run_python: Run a short Python script. Useful for extracting specific data from the source (sprite positions, level layouts, etc). Keep scripts simple — just parse and print.
- get_frame_region: Get exact pixel values for a region. Useful for inspecting what's at a specific location.
- get_action_history: See recent actions and whether they changed the frame (helps detect walls/blocked moves).
- submit_actions: Queue a sequence of actions to execute.

HOW TO PLAY:
1. Read the source code to understand the game mechanics and win condition.
2. Look at the frame to understand the current state.
3. Decide what actions to take based on your understanding.
4. Submit a sequence of actions, or output ACTION: <N> for a single move.

TIPS:
- The source code tells you everything: what each action does, what sprites mean, how to win.
- For movement games: identify your player, the goal, and obstacles. Navigate around obstacles.
- For click games: identify what to click and where. Output click coordinates with submit_actions.
- You can submit short action sequences (5-20 moves) and observe the result, then plan more.
- If an action didn't change the frame, you probably hit a wall — try a different direction."""


def build_step_message(game_id: str, obs, frame_text: str, diff_text: str,
                       memory_text: str, step_count: int) -> str:
    return f"""\
GAME: {game_id} | Level: {obs.levels_completed + 1}/{obs.win_levels} | Step: {step_count}
State: {obs.game_state} | Available actions: {obs.available_actions}

FRAME (16x16 downsampled, hex colors 0-F):
{frame_text}

CHANGES: {diff_text}
{memory_text}

What's your next move? If you haven't read the source yet, start with read_game_source."""
