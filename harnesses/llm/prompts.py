"""
Prompt templates for the LLM game agent.
"""

SYSTEM_PROMPT = """\
You are an agent playing an ARC-AGI-3 interactive puzzle game. You see a 64x64 pixel grid and choose actions each turn.

ACTIONS: 1=up, 2=down, 3=left, 4=right

GAME RULES:
- Games have multiple levels of increasing difficulty
- You have a limited step counter per life (moves are precious!)
- You have 3 lives per game-over; losing all lives resets to level 1
- Health pickups (if present) refill your step counter

TOOLS:
You have tools to help you analyze the game BEFORE acting:
- read_game_source: Read the game's Python source code to understand ALL mechanics
- run_python: Execute Python code to extract level data, compute paths, etc.
- get_frame_region: Inspect a specific region of the current frame in detail
- get_action_history: Review your recent actions and their outcomes
- submit_actions: Queue a sequence of actions to execute (use after computing a plan)

STRATEGY:
1. On the FIRST step of a new game, ALWAYS call read_game_source first
2. Then call run_python to extract the current level's layout: walls, doors, changers, pickups, player position, win conditions
3. Use run_python to compute the optimal action sequence (BFS/pathfinding)
4. Call submit_actions with the computed path
5. After level completion, repeat steps 2-4 for the next level

IMPORTANT: Use your tools to compute solutions. Do NOT guess moves blindly.
When you need to output a single action directly, respond with EXACTLY:
ACTION: <number>

Think step by step. Analyze before acting."""


def build_step_message(game_id: str, obs, frame_text: str, diff_text: str,
                       memory_text: str, step_count: int) -> str:
    """Build the user message for each game step."""
    return f"""\
GAME: {game_id} | Level: {obs.levels_completed + 1}/{obs.win_levels} | Step: {step_count}
State: {obs.game_state}

FRAME (downsampled 16x16, color codes: K=black B=blue R=red G=green Y=yellow W=gray M=magenta O=orange C=cyan D=dark-red I=indigo L=lime T=teal P=purple S=light-gray X=white):
{frame_text}

CHANGES: {diff_text}

{memory_text}

What is your next action? Use tools to analyze, or respond with ACTION: <1-4>"""
