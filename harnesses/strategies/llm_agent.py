"""
LLM-based game agent strategy.

Uses a local LLM (via ollama or compatible API) to:
1. Read and analyze the game's source code
2. Extract level layouts and compute plans via Python tool calls
3. Execute computed action sequences
4. Adapt when plans fail

The LLM does the reasoning at test time — no pre-computed solutions.
"""

import re
import random
from typing import Optional

from arcengine import GameAction

from ..base import Strategy
from ..perception import GameObservation
from ..llm.client import create_client, LLMResponse
from ..llm.tools import ToolExecutor, TOOL_DEFINITIONS
from ..llm.frame_encoder import encode_downsampled, encode_diff
from ..llm.prompts import SYSTEM_PROMPT, build_step_message

_ACTION_BY_ID = {a.value: a for a in GameAction}


class LLMAgentStrategy(Strategy):
    """
    LLM agent that reads game source code, analyzes mechanics,
    computes optimal paths, and executes them — all at test time.
    """

    name = "llm_agent"

    def __init__(
        self,
        action_space: list[int] = None,
        model: str = "qwen2.5:32b",
        backend: str = "ollama",
        base_url: str = "http://localhost:11434",
        max_tool_rounds: int = 10,
        temperature: float = 0.1,
        game_id: str = "ls20",
        dataset_dir: str = "dataset",
        **kwargs,
    ):
        self.action_space = action_space or [1, 2, 3, 4]
        self.client = create_client(backend, model, base_url)
        self.tools = ToolExecutor(game_id=game_id, dataset_dir=dataset_dir)
        self.max_tool_rounds = max_tool_rounds
        self.temperature = temperature
        self._step_count = 0
        self._level_step_count = 0
        self._current_level = -1
        self._conversation: list[dict] = []
        self._game_analyzed = False
        # Cache solutions per level so LLM only solves each level once
        self._level_solutions: dict[int, list[int]] = {}

    def reset(self):
        """New level — keep game analysis, reset level state."""
        self._level_step_count = 0
        self._conversation = []
        self.tools._pending_actions = []

    def soft_reset(self):
        """Death/game-over — reload cached solution if available."""
        self._level_step_count = 0
        self.tools._pending_actions = []
        # Reload cached solution for current level
        if self._current_level in self._level_solutions:
            self.tools._pending_actions = list(
                self._level_solutions[self._current_level]
            )

    def choose_action(self, obs: GameObservation) -> GameAction:
        self._step_count += 1
        self._level_step_count += 1

        # Detect level change
        if obs.levels_completed != self._current_level:
            if self._current_level >= 0:
                self.reset()
            self._current_level = obs.levels_completed

        # If we have queued actions (from submit_actions tool), use them
        if self.tools.has_pending_actions():
            action_id = self.tools.pop_action()
            if action_id is not None:
                return _ACTION_BY_ID[action_id]

        # Check if we have a cached solution for this level
        if self._current_level in self._level_solutions:
            self.tools._pending_actions = list(
                self._level_solutions[self._current_level]
            )
            action_id = self.tools.pop_action()
            if action_id is not None:
                return _ACTION_BY_ID[action_id]

        # Need LLM reasoning — build prompt and call
        self.tools.current_obs = obs
        action = self._llm_decide(obs)
        return _ACTION_BY_ID.get(action, _ACTION_BY_ID[random.choice(self.action_space)])

    def on_step_result(self, action: GameAction, obs: GameObservation):
        frame_changed = (obs.diff_from_prev is not None and
                         obs.diff_from_prev.changed)
        self.tools.update(obs, action.value, frame_changed)

    def get_level_solution(self, level_index: int) -> Optional[list[int]]:
        return list(self._level_solutions.get(level_index, []))

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "steps": self._step_count,
            "levels_solved": list(self._level_solutions.keys()),
            "conversation_length": len(self._conversation),
        }

    def _llm_decide(self, obs: GameObservation) -> int:
        """Run the LLM agent loop: tool calls until an action is decided."""
        # Build frame representation
        frame_text = encode_downsampled(obs.frame, factor=4,
                                        mask_rows=tuple(range(60, 64)))
        diff_text = "First observation" if obs.diff_from_prev is None else \
            encode_diff(obs.diff_from_prev, obs.frame)

        # Memory context
        memory_parts = []
        if self._game_analyzed:
            memory_parts.append("Game source has been read and analyzed.")
        if self._current_level in self._level_solutions:
            memory_parts.append(f"Level {self._current_level + 1} solution is cached ({len(self._level_solutions[self._current_level])} actions).")
        memory_text = " ".join(memory_parts) if memory_parts else "No analysis done yet. Start by reading the game source."

        user_msg = build_step_message(
            game_id=self.tools.game_id, obs=obs,
            frame_text=frame_text, diff_text=diff_text,
            memory_text=memory_text, step_count=self._step_count,
        )

        # Build conversation
        if not self._conversation:
            self._conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._conversation.append({"role": "user", "content": user_msg})

        # Keep conversation manageable — aggressive truncation for local models
        if len(self._conversation) > 16:
            self._conversation = [self._conversation[0]] + self._conversation[-8:]

        # LLM tool-calling loop
        for _round in range(self.max_tool_rounds):
            print(f"    [llm] round {_round + 1}...", end="", flush=True)
            try:
                response = self.client.chat(
                    self._conversation,
                    tools=TOOL_DEFINITIONS,
                    temperature=self.temperature,
                )
            except Exception as e:
                print(f" error: {e}")
                break

            # Handle tool calls
            if response.tool_calls:
                # Add assistant message with tool calls
                self._conversation.append({
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in response.tool_calls
                    ],
                })

                for tc in response.tool_calls:
                    args_preview = ""
                    if tc.name == "run_python":
                        code = tc.arguments.get("code", "")
                        args_preview = f" ({len(code)} chars)"
                    elif tc.name == "submit_actions":
                        actions = tc.arguments.get("actions", [])
                        args_preview = f" ({len(actions)} actions)"
                    print(f" {tc.name}{args_preview}", end="", flush=True)
                    result = self.tools.execute(tc.name, tc.arguments)

                    # Log tool results briefly
                    if tc.name == "run_python" and result.strip():
                        preview = result.strip()[:120].replace('\n', ' ')
                        print(f"\n      -> {preview}", end="", flush=True)

                    # Track special tool effects
                    if tc.name == "read_game_source":
                        self._game_analyzed = True
                    if tc.name == "submit_actions":
                        # Cache the submitted actions as this level's solution
                        actions = tc.arguments.get("actions", [])
                        if actions:
                            self._level_solutions[self._current_level] = list(actions)

                    self._conversation.append({
                        "role": "tool",
                        "name": tc.name,
                        "content": result[:8000],  # cap tool output
                    })

                print()

                # If submit_actions was called, use the queued actions
                if self.tools.has_pending_actions():
                    action_id = self.tools.pop_action()
                    if action_id is not None:
                        return action_id
                continue

            # No tool calls — parse action from text
            print(f" -> {response.text[:80]}")
            self._conversation.append({
                "role": "assistant", "content": response.text
            })
            action = self._parse_action(response.text)
            if action is not None:
                return action

            # LLM didn't output a clear action — ask again
            self._conversation.append({
                "role": "user",
                "content": "Please respond with ACTION: <1-4> or use a tool.",
            })

        # Fallback
        print("    [llm] fallback to random")
        return random.choice(self.action_space)

    def _parse_action(self, text: str) -> Optional[int]:
        """Extract action number from LLM text response."""
        # Look for ACTION: N pattern
        match = re.search(r'ACTION:\s*(\d+)', text, re.IGNORECASE)
        if match:
            action = int(match.group(1))
            if action in self.action_space:
                return action
        # Try bare number at end of response
        match = re.search(r'\b([1-4])\s*$', text.strip())
        if match:
            return int(match.group(1))
        return None
