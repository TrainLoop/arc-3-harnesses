"""
Base harness: abstract interface that connects a Strategy to a game environment.

A Harness is the complete package: perception + strategy + game loop.
Designed to be serializable so a model can learn to synthesize new harnesses.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from arcengine import GameAction, GameState

from .perception import Frame, GameObservation

_ACTION_BY_ID = {a.value: a for a in GameAction}


@dataclass
class HarnessConfig:
    """Serializable configuration for a harness. This is the 'representation'
    a model would learn to produce."""
    game_id: str
    strategy_name: str
    max_actions: int = 1000
    max_actions_per_level: int = 500
    action_space: list = field(default_factory=lambda: [1, 2, 3, 4])
    mask_rows: list = field(default_factory=list)  # rows to mask from state hash
    strategy_params: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "HarnessConfig":
        return cls(**json.loads(s))

    def save(self, path: str):
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "HarnessConfig":
        return cls.from_json(Path(path).read_text())


@dataclass
class ActionRecord:
    """One step in a game trajectory."""
    step: int
    action: int
    action_name: str
    state_hash_before: str
    state_hash_after: str
    game_state: str
    levels_completed: int
    frame_changed: bool


@dataclass
class HarnessResult:
    """Complete result of running a harness on a game."""
    game_id: str
    strategy_name: str
    total_actions: int
    levels_completed: int
    win_levels: int
    won: bool
    trajectory: list  # List[ActionRecord]
    duration_seconds: float

    def to_json(self) -> str:
        return json.dumps({
            "game_id": self.game_id,
            "strategy_name": self.strategy_name,
            "total_actions": self.total_actions,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "won": self.won,
            "duration_seconds": self.duration_seconds,
            "trajectory_length": len(self.trajectory),
        }, indent=2)


class Strategy(ABC):
    """Abstract action-selection strategy. Pluggable into any harness."""

    name: str = "base"

    @abstractmethod
    def reset(self):
        """Reset strategy state for a new level or game."""
        ...

    @abstractmethod
    def choose_action(self, obs: GameObservation) -> GameAction:
        """Given current observation, return the next action."""
        ...

    @abstractmethod
    def on_step_result(self, action: GameAction, obs: GameObservation):
        """Callback after an action is executed. Update internal state."""
        ...

    def get_shortest_path(self, from_hash: str, to_hash: str) -> Optional[list[int]]:
        """Get shortest action path between two states. Override in graph-based strategies."""
        return None

    def get_initial_hash(self) -> Optional[str]:
        """Get the initial state hash. Override in strategies that track this."""
        return None

    def get_level_solution(self, level_index: int) -> Optional[list[int]]:
        """Get pre-computed solution for a level. Override in source-aware strategies."""
        return None

    def serialize(self) -> dict:
        """Serialize strategy state for reproducibility / model training."""
        return {"name": self.name}


class Harness:
    """Connects a Strategy to a game environment and runs the game loop."""

    def __init__(self, config: HarnessConfig, strategy: Strategy):
        self.config = config
        self.strategy = strategy

    def run(self, env) -> HarnessResult:
        """Run the full game loop. `env` is an arc_agi EnvironmentWrapper.

        Key optimization: caches winning action sequences for completed levels.
        After a game-over reset (which goes back to level 1), replays cached
        paths to skip solved levels instantly.
        """
        start = time.time()
        trajectory = []
        total_actions = 0
        level_actions = 0
        prev_frame = None
        prev_levels = 0
        mask_rows = tuple(self.config.mask_rows)

        # Cache of winning action sequences per level (shortest known paths)
        level_paths: dict[int, list[int]] = {}  # level_index -> [action_ids]
        level_initial_hashes: dict[int, str] = {}  # level_index -> initial hash
        level_strategies: dict[int, Strategy] = {}  # level_index -> strategy (for path lookup)
        current_level_index: int = 0
        level_initial_hash: str = ""

        # Initial observation
        obs_raw = env.reset()
        obs = GameObservation.from_raw(obs_raw, mask_rows=mask_rows)
        prev_frame = obs.frame
        level_initial_hash = obs.state_hash
        level_initial_hashes[0] = obs.state_hash
        self.strategy.reset()

        while total_actions < self.config.max_actions:
            # Check if we need to replay cached paths for solved levels
            if obs.levels_completed < prev_levels and obs.levels_completed in level_paths:
                # We reset — replay known paths for levels we've already solved
                pass  # handled below in game_over section

            action = self.strategy.choose_action(obs)

            state_before = obs.state_hash
            obs_raw = env.step(action)
            if obs_raw is None:
                break

            obs = GameObservation.from_raw(obs_raw, prev_frame, mask_rows=mask_rows)
            total_actions += 1
            level_actions += 1

            record = ActionRecord(
                step=total_actions,
                action=action.value,
                action_name=action.name,
                state_hash_before=state_before,
                state_hash_after=obs.state_hash,
                game_state=obs.game_state,
                levels_completed=obs.levels_completed,
                frame_changed=obs.diff_from_prev.changed if obs.diff_from_prev else True,
            )
            trajectory.append(record)

            self.strategy.on_step_result(action, obs)
            prev_frame = obs.frame

            # Level advanced — extract shortest path from graph and cache it
            if obs.levels_completed > prev_levels:
                completed_level = prev_levels
                # Try to get optimal path: first from pre-computed solutions,
                # then from graph BFS, then empty fallback
                opt_path = self.strategy.get_level_solution(completed_level)
                if not opt_path:
                    init_hash = level_initial_hashes.get(completed_level)
                    if init_hash:
                        opt_path = self.strategy.get_shortest_path(init_hash, state_before)
                if opt_path:
                    level_paths[completed_level] = opt_path
                else:
                    level_paths[completed_level] = []

                # Save strategy for this level (for future path lookups)
                level_strategies[completed_level] = self.strategy

                prev_levels = obs.levels_completed
                current_level_index = obs.levels_completed
                level_actions = 0
                level_initial_hash = obs.state_hash
                level_initial_hashes[current_level_index] = obs.state_hash
                self.strategy.reset()
                print(f"  [level {obs.levels_completed}/{obs.win_levels}] "
                      f"completed at step {total_actions} "
                      f"(optimal path: {len(level_paths[completed_level])} actions)")

            # Game over — reset and replay cached level paths
            if obs.game_state == "GAME_OVER":
                obs_raw = env.reset()
                if obs_raw is None:
                    break
                obs = GameObservation.from_raw(obs_raw, prev_frame, mask_rows=mask_rows)
                prev_frame = obs.frame

                # Replay cached paths for all previously solved levels
                replay_target = prev_levels
                for lvl in range(replay_target):
                    if lvl not in level_paths:
                        break
                    path = level_paths[lvl]
                    for aid in path:
                        obs_raw = env.step(_ACTION_BY_ID[aid])
                        if obs_raw is None:
                            break
                        obs = GameObservation.from_raw(obs_raw, prev_frame, mask_rows=mask_rows)
                        total_actions += 1
                        prev_frame = obs.frame

                        if obs.game_state == "GAME_OVER":
                            break  # Replay path failed (shouldn't happen)
                    if obs_raw is None or obs.game_state == "GAME_OVER":
                        break

                if obs.game_state == "GAME_OVER":
                    # Replay failed — try again from scratch
                    obs_raw = env.reset()
                    if obs_raw is None:
                        break
                    obs = GameObservation.from_raw(obs_raw, prev_frame, mask_rows=mask_rows)
                    prev_frame = obs.frame

                level_initial_hash = obs.state_hash
                if hasattr(self.strategy, 'soft_reset'):
                    self.strategy.soft_reset()
                else:
                    self.strategy.reset()
                level_actions = 0

            # Win
            if obs.game_state == "WIN":
                print(f"  [WIN] at step {total_actions}")
                break

            # Per-level budget — trigger game over handling on next iteration
            if level_actions >= self.config.max_actions_per_level:
                print(f"  [budget] level budget exhausted at step {total_actions}")
                obs_raw = env.reset()
                if obs_raw is None:
                    break
                obs = GameObservation.from_raw(obs_raw, prev_frame, mask_rows=mask_rows)
                prev_frame = obs.frame

                # Replay cached levels
                for lvl in range(prev_levels):
                    if lvl not in level_paths:
                        break
                    for aid in level_paths[lvl]:
                        obs_raw = env.step(_ACTION_BY_ID[aid])
                        if obs_raw is None:
                            break
                        obs = GameObservation.from_raw(obs_raw, prev_frame, mask_rows=mask_rows)
                        total_actions += 1
                        prev_frame = obs.frame

                if hasattr(self.strategy, 'soft_reset'):
                    self.strategy.soft_reset()
                else:
                    self.strategy.reset()
                level_actions = 0

        return HarnessResult(
            game_id=self.config.game_id,
            strategy_name=self.strategy.name,
            total_actions=total_actions,
            levels_completed=obs.levels_completed,
            win_levels=obs.win_levels,
            won=obs.game_state == "WIN",
            trajectory=trajectory,
            duration_seconds=time.time() - start,
        )
