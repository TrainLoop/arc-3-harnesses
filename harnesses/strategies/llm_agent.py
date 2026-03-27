"""
LLM-based game agent strategy.

Uses an LLM (local or API) to:
1. Read and analyze the game's source code
2. Extract level layouts and compute plans via Python tool calls
3. Execute computed action sequences
4. Adapt when plans fail

Works with ANY ARC-AGI-3 game — keyboard, click, or mixed action spaces.
"""

import re
import json
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
    Works with any action space (keyboard, click, mixed).
    """

    name = "llm_agent"

    def __init__(
        self,
        action_space: list[int] = None,
        model: str = "gpt-5.2",
        backend: str = "openai",
        base_url: str = "http://localhost:11434",
        max_tool_rounds: int = 25,
        temperature: float = 0.2,
        game_id: str = "ls20",
        dataset_dir: str = "dataset",
        **kwargs,
    ):
        self.action_space = action_space or [1, 2, 3, 4]
        self.client = create_client(backend, model, base_url, **kwargs)
        self.tools = ToolExecutor(game_id=game_id, dataset_dir=dataset_dir)
        self.max_tool_rounds = max_tool_rounds
        self.temperature = temperature
        self._step_count = 0
        self._level_step_count = 0
        self._current_level = -1
        self._conversation: list[dict] = []
        self._game_analyzed = False
        self._level_solutions: dict[int, list[int]] = {}
        self._level_clicks: dict[int, list[list[int]]] = {}

    def reset(self):
        self._level_step_count = 0
        self._conversation = []
        self.tools._pending_actions = []
        self.tools._pending_clicks = []

    def soft_reset(self):
        self._level_step_count = 0
        self.tools._pending_actions = []
        self.tools._pending_clicks = []
        if self._current_level in self._level_solutions:
            self.tools._pending_actions = list(self._level_solutions[self._current_level])
            self.tools._pending_clicks = list(self._level_clicks.get(self._current_level, []))

    def choose_action(self, obs: GameObservation) -> GameAction:
        self._step_count += 1
        self._level_step_count += 1

        if obs.levels_completed != self._current_level:
            if self._current_level >= 0:
                self.reset()
            self._current_level = obs.levels_completed

        # Queued actions from submit_actions tool
        if self.tools.has_pending_actions():
            action_id = self.tools.pop_action()
            if action_id is not None:
                # For click actions, attach coordinates
                if action_id == 6:
                    click = self.tools.pop_click()
                    if click:
                        self._last_click = click  # stored for env.step data
                return _ACTION_BY_ID.get(action_id, _ACTION_BY_ID[1])

        # Cached solution
        if self._current_level in self._level_solutions:
            self.tools._pending_actions = list(self._level_solutions[self._current_level])
            self.tools._pending_clicks = list(self._level_clicks.get(self._current_level, []))
            action_id = self.tools.pop_action()
            if action_id is not None:
                if action_id == 6:
                    click = self.tools.pop_click()
                    if click:
                        self._last_click = click
                return _ACTION_BY_ID.get(action_id, _ACTION_BY_ID[1])

        # LLM reasoning
        self.tools.current_obs = obs
        action = self._llm_decide(obs)
        return _ACTION_BY_ID.get(action, _ACTION_BY_ID[random.choice(self.action_space)])

    def get_click_data(self) -> Optional[dict]:
        """Get click coordinates for the last chosen action (if action 6)."""
        if hasattr(self, '_last_click') and self._last_click:
            click = self._last_click
            self._last_click = None
            return {"x": click[0], "y": click[1]}
        return None

    def get_reasoning(self) -> Optional[dict]:
        """Get reasoning for the last chosen action (sent to API for logging).
        Includes the full LLM thought chain: tool calls, results, and thoughts."""
        parts = []
        # Walk conversation from the last user message to capture the full reasoning chain
        start_idx = 0
        for i in range(len(self._conversation) - 1, -1, -1):
            if self._conversation[i].get("role") == "user":
                start_idx = i + 1
                break

        for msg in self._conversation[start_idx:]:
            role = msg.get("role", "")
            if role == "assistant":
                text = msg.get("content", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", tc)
                        name = fn.get("name", "?")
                        args = fn.get("arguments", {})
                        if name == "run_python":
                            code = args.get("code", "")
                            parts.append(f"[tool: run_python]\n```\n{code[:300]}\n```")
                        elif name == "read_game_source":
                            parts.append("[tool: read_game_source]")
                        elif name == "submit_actions":
                            actions = args.get("actions", [])
                            reason = args.get("reasoning", "")
                            parts.append(f"[tool: submit_actions] {len(actions)} actions. {reason}")
                        else:
                            parts.append(f"[tool: {name}]")
            elif role == "tool":
                result = msg.get("content", "")[:200]
                parts.append(f"-> {result}")

        full_text = "\n".join(parts)
        # Cap at ~2KB to stay under 16KB reasoning limit
        if len(full_text) > 2000:
            full_text = full_text[:2000] + "\n..."
        return full_text if full_text else None

    def on_step_result(self, action: GameAction, obs: GameObservation):
        frame_changed = (obs.diff_from_prev is not None and obs.diff_from_prev.changed)
        self.tools.update(obs, action.value, frame_changed)

        # Loop detection: if the last N actions were identical and didn't change frame, clear the queue
        if not frame_changed:
            self._no_change_count = getattr(self, '_no_change_count', 0) + 1
        else:
            self._no_change_count = 0

        if self._no_change_count >= 5:
            # Stuck — clear cached solution and force re-reasoning
            self.tools._pending_actions = []
            self.tools._pending_clicks = []
            if self._current_level in self._level_solutions:
                del self._level_solutions[self._current_level]
            if self._current_level in self._level_clicks:
                del self._level_clicks[self._current_level]
            self._conversation = []  # reset conversation to avoid repeating
            self._no_change_count = 0
            print(f"    [loop-detect] cleared stale solution, will re-reason")

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
        frame_text = encode_downsampled(obs.frame, factor=4,
                                        mask_rows=tuple(range(60, 64)))
        diff_text = "First observation" if obs.diff_from_prev is None else \
            encode_diff(obs.diff_from_prev, obs.frame)

        memory_parts = []
        if self._game_analyzed:
            memory_parts.append("Game source has been read and analyzed.")
        if self._current_level in self._level_solutions:
            n = len(self._level_solutions[self._current_level])
            memory_parts.append(f"Level {self._current_level + 1} solution cached ({n} actions).")
        memory_text = " ".join(memory_parts) if memory_parts else \
            "No analysis done yet. Start by reading the game source."

        user_msg = build_step_message(
            game_id=self.tools.game_id, obs=obs,
            frame_text=frame_text, diff_text=diff_text,
            memory_text=memory_text, step_count=self._step_count,
        )

        if not self._conversation:
            self._conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._conversation.append({"role": "user", "content": user_msg})

        # Keep conversation manageable — but preserve tool call/result pairs
        if len(self._conversation) > 20:
            # Keep system + last N messages, ensuring no orphaned tool results
            keep = [self._conversation[0]]  # system
            tail = self._conversation[-12:]
            # If tail starts with a tool result, drop it (orphaned)
            while tail and tail[0].get("role") == "tool":
                tail = tail[1:]
            self._conversation = keep + tail

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

            if response.tool_calls:
                self._conversation.append({
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in response.tool_calls
                    ],
                })

                for tc in response.tool_calls:
                    preview = ""
                    if tc.name == "run_python":
                        preview = f" ({len(tc.arguments.get('code', ''))} chars)"
                    elif tc.name == "submit_actions":
                        preview = f" ({len(tc.arguments.get('actions', []))} actions)"
                    print(f" {tc.name}{preview}", end="", flush=True)
                    result = self.tools.execute(tc.name, tc.arguments)

                    if tc.name == "run_python" and result.strip():
                        lines = result.strip().split('\n')
                        show = lines[0][:120] if lines else ""
                        print(f"\n      -> {show}", end="", flush=True)

                    if tc.name == "read_game_source":
                        self._game_analyzed = True
                    if tc.name == "submit_actions":
                        actions = tc.arguments.get("actions", [])
                        clicks = tc.arguments.get("clicks", [])
                        if actions:
                            self._level_solutions[self._current_level] = list(actions)
                            if clicks:
                                self._level_clicks[self._current_level] = list(clicks)

                    self._conversation.append({
                        "role": "tool",
                        "name": tc.name,
                        "tool_use_id": tc.name,
                        "content": result[:8000],
                    })

                print()

                if self.tools.has_pending_actions():
                    action_id = self.tools.pop_action()
                    if action_id is not None:
                        if action_id == 6:
                            click = self.tools.pop_click()
                            if click:
                                self._last_click = click
                        return action_id
                continue

            print(f" -> {response.text[:80]}")
            self._conversation.append({"role": "assistant", "content": response.text})
            action = self._parse_action(response.text)
            if action is not None:
                return action

            self._conversation.append({
                "role": "user",
                "content": f"Please use tools or respond with ACTION: <N> where N is one of {self.action_space}",
            })

        print("    [llm] fallback to random")
        return random.choice(self.action_space)

    def _parse_action(self, text: str) -> Optional[int]:
        match = re.search(r'ACTION:\s*(\d+)', text, re.IGNORECASE)
        if match:
            action = int(match.group(1))
            if action in self.action_space:
                return action
        match = re.search(r'\b([1-7])\s*$', text.strip())
        if match:
            action = int(match.group(1))
            if action in self.action_space:
                return action
        return None
