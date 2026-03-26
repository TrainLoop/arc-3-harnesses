"""
Graph-based state exploration strategy with path replay.

Builds a directed graph where:
  - Nodes = unique game states (identified by frame hash)
  - Edges = (state, action) -> next_state transitions

Key insight for games with step limits (like ls20): after dying,
replay the shortest known path to the exploration frontier, then
continue exploring new states. This maximizes exploration per life.

Inspired by the 3rd-place ARC-AGI-3 solution (arXiv 2512.24156).
"""

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from arcengine import GameAction

from ..base import Strategy
from ..perception import GameObservation

# arcengine's GameAction doesn't support construction by int value
_ACTION_BY_ID = {a.value: a for a in GameAction}


def _action(action_id: int) -> GameAction:
    return _ACTION_BY_ID[action_id]


@dataclass
class StateNode:
    """A node in the state graph."""
    state_hash: str
    visit_count: int = 0
    transitions: dict = field(default_factory=dict)       # action_id -> state_hash
    action_productive: dict = field(default_factory=dict)  # action_id -> bool
    is_goal: bool = False
    depth: int = 0  # min steps from initial state

    def untested_actions(self, action_space: list[int]) -> list[int]:
        return [a for a in action_space if a not in self.transitions]

    def productive_actions(self) -> list[int]:
        return [a for a, p in self.action_productive.items() if p]


class StateGraph:
    """Directed graph of game states and transitions."""

    def __init__(self, action_space: list[int]):
        self.action_space = action_space
        self.nodes: dict[str, StateNode] = {}
        self.edge_count = 0

    def get_or_create(self, state_hash: str) -> StateNode:
        if state_hash not in self.nodes:
            self.nodes[state_hash] = StateNode(state_hash=state_hash)
        return self.nodes[state_hash]

    def record_transition(self, from_hash: str, action_id: int,
                          to_hash: str, productive: bool):
        from_node = self.get_or_create(from_hash)
        to_node = self.get_or_create(to_hash)
        from_node.transitions[action_id] = to_hash
        from_node.action_productive[action_id] = productive
        from_node.visit_count += 1
        self.edge_count += 1
        # Track depth (min distance from any known starting state)
        if to_node.depth == 0 and from_node.depth > 0:
            to_node.depth = from_node.depth + 1

    def bfs_path(self, from_hash: str, target_fn) -> Optional[list[int]]:
        """BFS to find shortest action path to a state matching target_fn."""
        if from_hash not in self.nodes:
            return None
        if target_fn(self.nodes[from_hash]):
            return []

        visited = {from_hash}
        queue = deque([(from_hash, [])])

        while queue:
            current_hash, path = queue.popleft()
            current = self.nodes[current_hash]

            for action_id, next_hash in current.transitions.items():
                if next_hash in visited or next_hash not in self.nodes:
                    continue
                visited.add(next_hash)
                new_path = path + [action_id]

                if target_fn(self.nodes[next_hash]):
                    return new_path
                queue.append((next_hash, new_path))

        return None

    def path_to_frontier(self, from_hash: str) -> Optional[list[int]]:
        """Find shortest path to a state with untested actions."""
        return self.bfs_path(
            from_hash,
            lambda n: len(n.untested_actions(self.action_space)) > 0
        )

    def path_to_deepest_frontier(self, from_hash: str) -> Optional[list[int]]:
        """Find path to the deepest (highest depth) frontier state."""
        # First find all frontier states reachable from current
        if from_hash not in self.nodes:
            return None

        visited = {from_hash}
        queue = deque([(from_hash, [])])
        best_path = None
        best_depth = -1

        while queue:
            current_hash, path = queue.popleft()
            current = self.nodes[current_hash]

            if current.untested_actions(self.action_space):
                if current.depth > best_depth:
                    best_depth = current.depth
                    best_path = path

            for action_id, next_hash in current.transitions.items():
                if next_hash in visited or next_hash not in self.nodes:
                    continue
                visited.add(next_hash)
                queue.append((next_hash, path + [action_id]))

        return best_path

    def stats(self) -> dict:
        total = len(self.nodes)
        frontier = sum(
            1 for n in self.nodes.values()
            if n.untested_actions(self.action_space)
        )
        max_depth = max((n.depth for n in self.nodes.values()), default=0)
        return {
            "nodes": total,
            "edges": self.edge_count,
            "frontier": frontier,
            "fully_explored": total - frontier,
            "max_depth": max_depth,
        }


class GraphExplorerStrategy(Strategy):
    """
    Graph-based exploration with path replay for step-limited games.

    After each death/reset:
    1. Replay shortest known path to deepest frontier state
    2. Explore untested actions at that state
    3. If no frontier reachable, try random productive actions to discover new states
    """

    name = "graph_explorer"

    def __init__(
        self,
        action_space: list[int] = None,
        explore_depth: int = 200,
        prefer_productive: bool = True,
    ):
        self.action_space = action_space or [1, 2, 3, 4]
        self.explore_depth = explore_depth
        self.prefer_productive = prefer_productive
        self.graph = StateGraph(self.action_space)
        self._nav_path: list[int] = []
        self._current_hash: str = ""
        self._step_count = 0
        self._initial_hash: str = ""
        self._lives_since_progress = 0
        self._max_lives_no_progress = 50

    def reset(self):
        """Full reset for a new level."""
        self.graph = StateGraph(self.action_space)
        self._nav_path = []
        self._current_hash = ""
        self._step_count = 0
        self._initial_hash = ""
        self._lives_since_progress = 0

    def soft_reset(self):
        """Reset navigation but keep graph (after death)."""
        self._nav_path = []
        self._current_hash = ""
        self._step_count = 0
        self._lives_since_progress += 1

    def choose_action(self, obs: GameObservation) -> GameAction:
        self._current_hash = obs.state_hash
        node = self.graph.get_or_create(obs.state_hash)
        self._step_count += 1

        if not self._initial_hash:
            self._initial_hash = obs.state_hash
            node.depth = 1

        # Phase 1: If we have a replay/navigation path, follow it
        if self._nav_path:
            action_id = self._nav_path.pop(0)
            return _action(action_id)

        # Phase 2: Try untested actions at current state
        untested = node.untested_actions(self.action_space)
        if untested:
            random.shuffle(untested)
            return _action(untested[0])

        # Phase 3: Navigate to deepest frontier (maximize exploration per life)
        path = self.graph.path_to_deepest_frontier(obs.state_hash)
        if path is not None and len(path) > 0:
            self._nav_path = path[1:]
            return _action(path[0])

        # Phase 4: All explored — do random walks biased toward
        # least-visited states to discover new states
        all_actions = self.action_space[:]
        random.shuffle(all_actions)

        # Prefer actions leading to least-visited states
        scored = []
        for a in all_actions:
            nh = node.transitions.get(a)
            if nh and nh in self.graph.nodes:
                scored.append((self.graph.nodes[nh].visit_count, a))
            else:
                scored.append((0, a))
        scored.sort()
        return _action(scored[0][1])

    def on_step_result(self, action: GameAction, obs: GameObservation):
        productive = (
            obs.diff_from_prev is not None
            and obs.diff_from_prev.changed
        )
        self.graph.record_transition(
            self._current_hash,
            action.value,
            obs.state_hash,
            productive,
        )

        # If we're navigating and state doesn't match expected, replan
        if self._nav_path and self._current_hash in self.graph.nodes:
            expected_node = self.graph.nodes[self._current_hash]
            if self._nav_path:
                next_act = self._nav_path[0]
                expected_next = expected_node.transitions.get(next_act)
                if expected_next and obs.state_hash != expected_next:
                    self._nav_path = []  # Path invalidated

    def get_shortest_path(self, from_hash: str, to_hash: str):
        return self.graph.bfs_path(from_hash, lambda n: n.state_hash == to_hash)

    def get_initial_hash(self):
        return self._initial_hash

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "action_space": self.action_space,
            "explore_depth": self.explore_depth,
            "prefer_productive": self.prefer_productive,
            "graph_stats": self.graph.stats(),
        }
