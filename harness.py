#!/usr/bin/env python3
"""
LLM Harness for ARC-AGI-3 Games
Demonstrates an LLM (Claude) analyzing source code to build a programmatic solver.
Uses BFS pathfinding + Claude API fallback for complex levels.
"""

import sys
import os
import json
import uuid
import time
import re
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple, Set, Optional, Any

import numpy as np

# ============================================================
# Action Constants
# ============================================================
UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
ACTION_DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
REVERSE_ACTION = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}


# ============================================================
# Game Loading
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
    raise ValueError("No game class found in source.py")


# ============================================================
# Changer Movement Simulation (mirrors dboxixicic logic)
# ============================================================
DIR_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 0=down, 1=right, 2=up, 3=left


def simulate_changer(boundary_sprite, start_x, start_y, cell, n_steps):
    """
    Simulate changer movement within a boundary sprite.
    Returns list of (x, y) positions for steps 0..n_steps.
    Position at index t is the changer's position AFTER t steps.
    """
    bx, by = boundary_sprite.x, boundary_sprite.y
    bw, bh = boundary_sprite.width, boundary_sprite.height
    pixels = boundary_sprite.pixels  # numpy array

    def is_valid(x, y):
        if x < bx or x >= bx + bw or y < by or y >= by + bh:
            return False
        px, py = x - bx, y - by
        return int(pixels[py, px]) >= 0

    x, y = start_x, start_y
    direction = 0
    positions = [(x, y)]

    for _ in range(n_steps):
        moved = False
        for d_off in [0, -1, 1, 2]:
            d = (direction + d_off) % 4
            dx, dy = DIR_DELTAS[d]
            nx, ny = x + dx * cell, y + dy * cell
            if is_valid(nx, ny):
                direction = d
                x, y = nx, ny
                moved = True
                break
        positions.append((x, y))

    return positions


# ============================================================
# Grid-Based BFS Solver
# ============================================================
class GridSolver:
    """Solves a level using BFS pathfinding derived from source code analysis."""

    def __init__(self, game):
        self.game = game
        self.pw = game.gisrhqpee  # player width (movement step X)
        self.ph = game.tbwnoxqgc  # player height (movement step Y)
        gs = game.current_level.grid_size or (64, 64)
        self.grid_w, self.grid_h = gs

        # Obstacles
        self.obstacles: Set[Tuple[int, int]] = set()
        for s in game.current_level.get_sprites_by_tag("ihdgageizm"):
            self.obstacles.add((s.x, s.y))

        # Player state (needed early for push block computation)
        self.start_pos = (game.gudziatsk.x, game.gudziatsk.y)

        # Goals
        self.goals = []
        for i, s in enumerate(game.plrpelhym):
            self.goals.append({
                "x": s.x, "y": s.y,
                "shape": game.ldxlnycps[i],
                "color_idx": game.yjdexjsoa[i],
                "rotation_idx": game.ehwheiwsk[i],
            })

        # Push blocks: model as teleportation (not obstacles)
        self.push_block_positions: Set[Tuple[int, int]] = set()
        self.push_teleports: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._compute_push_teleports(game)

        # Static changer positions
        self.changers: Dict[str, List[Tuple[int, int]]] = {}
        for tag, name in [("ttfwljgohq", "shape"), ("soyhouuebz", "color"), ("rhsxkxzdjz", "rotation")]:
            sprites = game.current_level.get_sprites_by_tag(tag)
            if sprites:
                self.changers[name] = [(s.x, s.y) for s in sprites]

        # Collectibles
        self.collectibles = [(s.x, s.y) for s in game.current_level.get_sprites_by_tag("npxgalaybz")]
        self.shape = game.fwckfzsyc
        self.color_idx = game.hiaauhahz
        self.rotation_idx = game.cklxociuu
        self.n_shapes = len(game.ijessuuig)
        self.n_colors = len(game.tnkekoeuk)

        # Step budget
        self.max_steps = game._step_counter_ui.osgviligwp
        self.decrement = game._step_counter_ui.efipnixsvl
        self.step_budget = self.max_steps // self.decrement

        # Moving changer simulations
        self.moving_changers: Dict[str, List[Tuple[int, int]]] = {}
        self._build_moving_changer_sims()

    def _build_moving_changer_sims(self):
        """Precompute positions for moving changers."""
        for ctrl in self.game.wsoslqeku:
            sprite = ctrl._sprite
            boundary = ctrl.bfdcztirdu
            if boundary and sprite.tags:
                for tag, name in [("ttfwljgohq", "shape"), ("soyhouuebz", "color"), ("rhsxkxzdjz", "rotation")]:
                    if tag in sprite.tags:
                        positions = simulate_changer(
                            boundary, sprite.x, sprite.y, ctrl._cell, 300
                        )
                        self.moving_changers[name] = positions
                        break

    def is_blocked(self, new_x: int, new_y: int, extra_obs: Optional[Set] = None) -> bool:
        """Check if position is blocked by obstacles."""
        all_obs = self.obstacles | self.push_block_positions
        if extra_obs:
            all_obs = all_obs | extra_obs
        for ox, oy in all_obs:
            if new_x <= ox < new_x + self.pw and new_y <= oy < new_y + self.ph:
                return True
        return False

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_w and 0 <= y < self.grid_h

    def bfs_path(self, start: Tuple[int, int], target: Tuple[int, int],
                 extra_obs: Optional[Set] = None) -> Optional[List[int]]:
        """BFS shortest path from start to target, avoiding obstacles."""
        sx, sy = start
        tx, ty = target
        if sx == tx and sy == ty:
            return []

        queue = deque([(sx, sy, [])])
        visited = {(sx, sy)}

        while queue:
            x, y, actions = queue.popleft()
            for action in [UP, DOWN, LEFT, RIGHT]:
                dc, dr = ACTION_DELTAS[action]
                nx, ny = x + dc * self.pw, y + dr * self.ph
                if not self.in_bounds(nx, ny):
                    continue
                if self.is_blocked(nx, ny, extra_obs):
                    continue
                # Push block teleportation
                if (nx, ny) in self.push_teleports:
                    nx, ny = self.push_teleports[(nx, ny)]
                    if not self.in_bounds(nx, ny):
                        continue
                if (nx, ny) in visited:
                    continue
                new_actions = actions + [action]
                if nx == tx and ny == ty:
                    return new_actions
                visited.add((nx, ny))
                queue.append((nx, ny, new_actions))
        return None

    def bfs_timed_budget(self, start: Tuple[int, int], changer_positions: List[Tuple[int, int]],
                         start_time: int, budget: int, coll_list: List[Tuple[int, int]],
                         coll_mask: int, obs: Optional[Set] = None,
                         max_time: int = 200, min_remaining: int = 0
                         ) -> Optional[Tuple[List[int], int, int, Tuple[int, int]]]:
        """BFS with time + budget + collectible support for moving changer interception.
        Returns (actions, remaining_budget, coll_mask, final_position) or None."""
        sx, sy = start
        coll_idx = {pos: i for i, pos in enumerate(coll_list)}
        best: Dict[Tuple, int] = {}
        init = (sx, sy, start_time, coll_mask)
        best[init] = budget
        queue = deque([(sx, sy, start_time, budget, coll_mask, [])])

        while queue:
            x, y, t, b, mask, actions = queue.popleft()
            if t >= max_time or t + 1 >= len(changer_positions):
                continue
            key = (x, y, t, mask)
            if best.get(key, -1) > b:
                continue

            for action in [UP, DOWN, LEFT, RIGHT]:
                dc, dr = ACTION_DELTAS[action]
                nx, ny = x + dc * self.pw, y + dr * self.ph
                nt = t + 1
                if not self.in_bounds(nx, ny) or self.is_blocked(nx, ny, obs):
                    nx, ny = x, y
                if (nx, ny) in self.push_teleports:
                    nx, ny = self.push_teleports[(nx, ny)]

                nb, nmask = b - 1, mask
                if (nx, ny) in coll_idx:
                    ci = coll_idx[(nx, ny)]
                    if not (mask & (1 << ci)):
                        nb = self.step_budget
                        nmask = mask | (1 << ci)
                if nb < 0:
                    continue

                cx, cy = changer_positions[nt]
                if nx <= cx < nx + self.pw and ny <= cy < ny + self.ph:
                    if nb >= min_remaining:
                        return (actions + [action], nb, nmask, (nx, ny))

                if nb <= min_remaining:
                    # Not enough budget to continue (need to find collectible)
                    pass  # still explore - might find collectible

                nkey = (nx, ny, nt, nmask)
                if nb > best.get(nkey, -1):
                    best[nkey] = nb
                    queue.append((nx, ny, nt, nb, nmask, actions + [action]))
        return None

    def bfs_timed(self, start: Tuple[int, int], changer_positions: List[Tuple[int, int]],
                  start_time: int = 0, extra_obs: Optional[Set] = None,
                  max_time: int = 150) -> Optional[List[int]]:
        """BFS with time dimension to intercept a moving changer."""
        sx, sy = start
        queue = deque([(sx, sy, start_time, [])])
        visited = {(sx, sy, start_time)}

        while queue:
            x, y, t, actions = queue.popleft()
            if t >= max_time or t + 1 >= len(changer_positions):
                continue

            for action in [UP, DOWN, LEFT, RIGHT]:
                dc, dr = ACTION_DELTAS[action]
                nx, ny = x + dc * self.pw, y + dr * self.ph
                nt = t + 1

                if not self.in_bounds(nx, ny) or self.is_blocked(nx, ny, extra_obs):
                    nx, ny = x, y  # blocked, stay in place

                # Check if changer at nt overlaps with player at (nx, ny)
                cx, cy = changer_positions[nt]
                if nx <= cx < nx + self.pw and ny <= cy < ny + self.ph:
                    return actions + [action]

                state = (nx, ny, nt)
                if state not in visited:
                    visited.add(state)
                    queue.append((nx, ny, nt, actions + [action]))
        return None

    def find_step_off_actions(self, pos: Tuple[int, int],
                              extra_obs: Optional[Set] = None) -> Optional[List[int]]:
        """Find 2 actions to step off a position and back on."""
        x, y = pos
        for action in [UP, DOWN, LEFT, RIGHT]:
            dc, dr = ACTION_DELTAS[action]
            nx, ny = x + dc * self.pw, y + dr * self.ph
            if self.in_bounds(nx, ny) and not self.is_blocked(nx, ny, extra_obs):
                return [action, REVERSE_ACTION[action]]
        return None

    def compute_position_after_actions(self, start: Tuple[int, int], actions: List[int],
                                       extra_obs: Optional[Set] = None) -> Tuple[int, int]:
        """Compute player position after executing actions."""
        x, y = start
        for action in actions:
            dc, dr = ACTION_DELTAS[action]
            nx, ny = x + dc * self.pw, y + dr * self.ph
            if self.in_bounds(nx, ny) and not self.is_blocked(nx, ny, extra_obs):
                if (nx, ny) in self.push_teleports:
                    nx, ny = self.push_teleports[(nx, ny)]
                x, y = nx, ny
        return (x, y)

    def _compute_push_teleports(self, game):
        """Compute push block teleportation: trigger_position -> destination."""
        # Build obstacle+goal set for push distance computation
        wall_set: Set[Tuple[int, int]] = set()
        for s in game.current_level.get_sprites_by_tag("ihdgageizm"):
            wall_set.add((s.x, s.y))
        for g in self.goals:
            wall_set.add((g["x"], g["y"]))

        origin_x = self.start_pos[0] % self.pw
        origin_y = self.start_pos[1] % self.ph

        for s in game.current_level.get_sprites_by_tag("gbvqrjtaqo"):
            if s.name.endswith("_t"):
                dx, dy = 0, -1
            elif s.name.endswith("_b"):
                dx, dy = 0, 1
            elif s.name.endswith("_r"):
                dx, dy = 1, 0
            elif s.name.endswith("_l"):
                dx, dy = -1, 0
            else:
                continue

            # Compute push distance (mirrors ullzqnksoj logic)
            wall_cx = s.x + dx
            wall_cy = s.y + dy
            push_dist = 0
            for dist in range(1, 12):
                check_x = wall_cx + dx * s.width * dist
                check_y = wall_cy + dy * s.height * dist
                if (check_x, check_y) in wall_set:
                    push_dist = max(0, dist - 1)
                    break
            if push_dist <= 0:
                continue

            disp_x = dx * s.width * push_dist
            disp_y = dy * s.height * push_dist

            # Compute trigger positions (player bbox overlaps push block bbox)
            for px in range(s.x - self.pw + 1, s.x + s.width):
                for py in range(s.y - self.ph + 1, s.y + s.height):
                    if ((px - origin_x) % self.pw == 0 and
                            (py - origin_y) % self.ph == 0 and
                            0 <= px < self.grid_w and 0 <= py < self.grid_h):
                        dest_x = px + disp_x
                        dest_y = py + disp_y
                        if 0 <= dest_x < self.grid_w and 0 <= dest_y < self.grid_h:
                            self.push_teleports[(px, py)] = (dest_x, dest_y)

    def _collection_pos(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        """Find the grid position where player would collect item at (cx, cy)."""
        origin_x = self.start_pos[0] % self.pw
        origin_y = self.start_pos[1] % self.ph
        gx = origin_x + ((cx - origin_x) // self.pw) * self.pw
        gy = origin_y + ((cy - origin_y) // self.ph) * self.ph
        if gx <= cx < gx + self.pw and gy <= cy < gy + self.ph:
            if self.in_bounds(gx, gy):
                return (gx, gy)
        return None

    def _goal_obstacles(self, exclude_idx: int = -1) -> Set[Tuple[int, int]]:
        obs = set()
        for j, g in enumerate(self.goals):
            if j != exclude_idx:
                obs.add((g["x"], g["y"]))
        return obs

    def bfs_multi_waypoint(self, start: Tuple[int, int],
                           waypoints: List[Tuple[int, int]],
                           budget: int,
                           coll_list: List[Tuple[int, int]],
                           coll_mask: int,
                           obs: Optional[Set] = None,
                           changer_avoid_after: int = -1,
                           changer_positions: Optional[Set] = None
                           ) -> Optional[Tuple[List[int], int, int]]:
        """
        Multi-waypoint BFS with budget and collectible support.
        Visits waypoints in order. Consecutive same-position waypoints
        require leaving and returning (for changer revisits).
        After changer_avoid_after waypoints, changer_positions are obstacles.
        Returns (actions, remaining_budget, coll_mask) or None.
        """
        n_wp = len(waypoints)
        if n_wp == 0:
            return ([], budget, coll_mask)

        # leave_required[i] = True if must leave waypoints[i] before next
        leave_req = []
        for i in range(n_wp - 1):
            leave_req.append(waypoints[i] == waypoints[i + 1])
        leave_req.append(False)

        coll_idx = {pos: i for i, pos in enumerate(coll_list)}
        changer_set = changer_positions or set()

        # State: (x, y, budget, coll_mask, wp_idx, must_leave)
        # Optimization: track best budget per (x, y, coll_mask, wp_idx, must_leave)
        best_budget: Dict[Tuple, int] = {}

        sx, sy = start
        init_state = (sx, sy, budget, coll_mask, 0, False)
        queue = deque([(init_state, [])])
        key = (sx, sy, coll_mask, 0, False)
        best_budget[key] = budget

        while queue:
            state, actions = queue.popleft()
            x, y, b, mask, wp, ml = state

            # Prune: if we've seen this position with better budget, skip
            key = (x, y, mask, wp, ml)
            if best_budget.get(key, -1) > b:
                continue

            for action in [UP, DOWN, LEFT, RIGHT]:
                dc, dr = ACTION_DELTAS[action]
                nx, ny = x + dc * self.pw, y + dr * self.ph
                if not self.in_bounds(nx, ny):
                    continue
                if self.is_blocked(nx, ny, obs):
                    continue

                # Changer avoidance after all changer visits
                extra = changer_set if wp >= changer_avoid_after and changer_avoid_after >= 0 else set()
                if (nx, ny) in extra:
                    continue

                # Push block teleportation
                if (nx, ny) in self.push_teleports:
                    nx, ny = self.push_teleports[(nx, ny)]
                    if not self.in_bounds(nx, ny):
                        continue

                nb, nmask = b - 1, mask
                if (nx, ny) in coll_idx:
                    ci = coll_idx[(nx, ny)]
                    if not (mask & (1 << ci)):
                        nb = self.step_budget
                        nmask = mask | (1 << ci)
                if nb < 0:
                    continue

                new_ml = ml
                new_wp = wp

                # Check if leaving current waypoint
                if ml and new_wp > 0 and (nx, ny) != waypoints[new_wp - 1]:
                    new_ml = False

                # Check if reaching next waypoint
                if not new_ml and new_wp < n_wp and (nx, ny) == waypoints[new_wp]:
                    new_wp += 1
                    if new_wp < n_wp and leave_req[new_wp - 1]:
                        new_ml = True
                    if new_wp >= n_wp:
                        return (actions + [action], nb, nmask)

                nkey = (nx, ny, nmask, new_wp, new_ml)
                if nb > best_budget.get(nkey, -1):
                    best_budget[nkey] = nb
                    queue.append(((nx, ny, nb, nmask, new_wp, new_ml), actions + [action]))

        return None

    def solve_level(self) -> Optional[List[int]]:
        """Plan action sequence solving the level with multi-waypoint BFS."""
        if not self.goals:
            return []

        coll_list: List[Tuple[int, int]] = []
        for cx, cy in self.collectibles:
            cp = self._collection_pos(cx, cy)
            if cp:
                coll_list.append(cp)

        all_actions: List[int] = []
        pos = self.start_pos
        budget = self.step_budget
        coll_mask = 0
        current_shape = self.shape
        current_color = self.color_idx
        current_rotation = self.rotation_idx
        current_time = 0

        for gi, goal in enumerate(self.goals):
            shape_steps = (goal["shape"] - current_shape) % self.n_shapes
            color_steps = (goal["color_idx"] - current_color) % self.n_colors
            rotation_steps = (goal["rotation_idx"] - current_rotation) % 4

            changer_visits: List[Tuple[str, int]] = []
            if shape_steps > 0 and "shape" in self.changers:
                changer_visits.append(("shape", shape_steps))
            if color_steps > 0 and "color" in self.changers:
                changer_visits.append(("color", color_steps))
            if rotation_steps > 0 and "rotation" in self.changers:
                changer_visits.append(("rotation", rotation_steps))

            goal_obs = self._goal_obstacles(gi)
            goal_pos = (goal["x"], goal["y"])

            # Check for moving changers - visit them FIRST for maximum budget
            has_moving = any(ct in self.moving_changers for ct, _ in changer_visits)
            if has_moving:
                moving_first = [cv for cv in changer_visits if cv[0] in self.moving_changers]
                static_after = [cv for cv in changer_visits if cv[0] not in self.moving_changers]
                changer_visits = moving_first + static_after

            if has_moving:
                # Handle each changer separately, goal at the end
                all_changer_positions: Set[Tuple[int, int]] = set()
                for ct, nv in changer_visits:
                    if ct not in self.moving_changers and ct in self.changers:
                        all_changer_positions.add(self.changers[ct][0])

                for changer_type, n_visits in changer_visits:
                    if changer_type in self.moving_changers:
                        positions = self.moving_changers[changer_type]
                        for visit in range(n_visits):
                            remaining_stepoffs = max(0, n_visits - visit - 1)
                            reserved = remaining_stepoffs * 2
                            result = self.bfs_timed_budget(
                                pos, positions, current_time, budget,
                                coll_list, coll_mask, goal_obs,
                                min_remaining=reserved)
                            if result is None:
                                return None
                            actions, budget, coll_mask, new_pos = result
                            all_actions.extend(actions)
                            current_time += len(actions)
                            pos = new_pos
                            if visit < n_visits - 1:
                                step_off = self.find_step_off_actions(pos, goal_obs)
                                if step_off is None:
                                    return None
                                all_actions.extend(step_off)
                                budget -= len(step_off)
                                current_time += len(step_off)
                    else:
                        changer_pos = self.changers[changer_type][0]
                        wp = [changer_pos] * n_visits
                        result = self.bfs_multi_waypoint(
                            pos, wp, budget, coll_list, coll_mask, goal_obs)
                        if result is None:
                            return None
                        actions, budget, coll_mask = result
                        all_actions.extend(actions)
                        current_time += len(actions)
                        pos = changer_pos

                    if changer_type == "shape":
                        current_shape = (current_shape + n_visits) % self.n_shapes
                    elif changer_type == "color":
                        current_color = (current_color + n_visits) % self.n_colors
                    elif changer_type == "rotation":
                        current_rotation = (current_rotation + n_visits) % 4

                # Navigate to goal after all changers
                result = self.bfs_multi_waypoint(
                    pos, [goal_pos], budget, coll_list, coll_mask, goal_obs,
                    changer_avoid_after=0, changer_positions=all_changer_positions)
                if result is None:
                    return None
                actions, budget, coll_mask = result
                all_actions.extend(actions)
                current_time += len(actions)
                pos = goal_pos
            else:
                # Build full waypoint list: all changer visits + goal
                waypoints: List[Tuple[int, int]] = []
                changer_positions_set: Set[Tuple[int, int]] = set()
                n_changer_wps = 0

                for changer_type, n_visits in changer_visits:
                    changer_pos = self.changers[changer_type][0]
                    changer_positions_set.add(changer_pos)
                    for _ in range(n_visits):
                        waypoints.append(changer_pos)
                    n_changer_wps += n_visits

                waypoints.append(goal_pos)

                result = self.bfs_multi_waypoint(
                    pos, waypoints, budget, coll_list, coll_mask, goal_obs,
                    changer_avoid_after=n_changer_wps,
                    changer_positions=changer_positions_set)
                if result is None:
                    return None

                actions, budget, coll_mask = result
                all_actions.extend(actions)
                current_time += len(actions)
                pos = goal_pos

                for changer_type, n_visits in changer_visits:
                    if changer_type == "shape":
                        current_shape = (current_shape + n_visits) % self.n_shapes
                    elif changer_type == "color":
                        current_color = (current_color + n_visits) % self.n_colors
                    elif changer_type == "rotation":
                        current_rotation = (current_rotation + n_visits) % 4

        return all_actions


# ============================================================
# LLM Solver (Claude API fallback)
# ============================================================
class LLMSolver:
    """Fallback solver using Claude API for complex levels."""

    def __init__(self, source_code: str):
        self.client = None
        self.source_code = source_code
        try:
            import anthropic
            self.client = anthropic.Anthropic()
        except Exception:
            pass

    def solve(self, state_desc: str) -> Optional[List[int]]:
        if not self.client:
            return None

        prompt = f"""You are solving an ARC-AGI-3 puzzle game. Analyze the source code and game state to produce an optimal action sequence.

GAME SOURCE CODE (key mechanics):
- Actions: 1=UP(-Y), 2=DOWN(+Y), 3=LEFT(-X), 4=RIGHT(+X)
- Player moves by 5 pixels per step
- Obstacles (ihdgageizm) block movement
- Shape changer (ttfwljgohq): cycles shape index +1 mod 6
- Color changer (soyhouuebz): cycles color index +1 mod 4
- Rotation changer (rhsxkxzdjz): cycles rotation index +1 mod 4
- Collectible (npxgalaybz): refills step counter to max
- Goal (rjlbuycveu): must stand on it with EXACT matching shape/color/rotation
- Wrong attributes on goal = BLOCKED + penalty
- Win = all goals satisfied

FULL SOURCE CODE:
```python
{self.source_code}
```

CURRENT STATE:
{state_desc}

INSTRUCTIONS:
1. Analyze the maze layout from obstacle positions
2. Determine which changers to visit and how many times
3. Plan the shortest path: player -> changers -> goal
4. Account for step budget

Return ONLY a JSON array of action numbers (1-4). Example: [3, 3, 1, 1, 4, 4, 2]"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            match = re.search(r'\[[\d\s,]+\]', text)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"    LLM error: {e}")
        return None


# ============================================================
# State Description Builder (for LLM)
# ============================================================
def build_state_description(game) -> str:
    """Build human-readable state description for LLM solver."""
    g = game
    pw, ph = g.gisrhqpee, g.tbwnoxqgc
    lines = []
    lines.append(f"Level: {g._current_level_index + 1} / {len(g._levels)}")
    lines.append(f"Player: ({g.gudziatsk.x}, {g.gudziatsk.y}), size {pw}x{ph}")
    lines.append(f"Shape: {g.fwckfzsyc} (of {len(g.ijessuuig)})")
    lines.append(f"Color index: {g.hiaauhahz} (of {len(g.tnkekoeuk)})")
    lines.append(f"Rotation index: {g.cklxociuu} (rotations: 0,90,180,270)")
    lines.append(f"Steps: {g._step_counter_ui.current_steps}/{g._step_counter_ui.osgviligwp}, decrement: {g._step_counter_ui.efipnixsvl}")
    lines.append(f"Lives: {g.aqygnziho}")

    lines.append("\nGoals:")
    for i, s in enumerate(g.plrpelhym):
        lines.append(f"  [{i}] pos=({s.x},{s.y}) need: shape={g.ldxlnycps[i]}, color_idx={g.yjdexjsoa[i]}, rot_idx={g.ehwheiwsk[i]}")

    lines.append("\nChangers:")
    for tag, name in [("ttfwljgohq", "shape"), ("soyhouuebz", "color"), ("rhsxkxzdjz", "rotation")]:
        for s in g.current_level.get_sprites_by_tag(tag):
            moving = " [MOVING]" if any(tag in (c._sprite.tags or []) for c in g.wsoslqeku) else ""
            lines.append(f"  {name}: ({s.x},{s.y}){moving}")

    if g.current_level.get_sprites_by_tag("npxgalaybz"):
        lines.append("\nCollectibles (step refills):")
        for s in g.current_level.get_sprites_by_tag("npxgalaybz"):
            lines.append(f"  ({s.x},{s.y})")

    # Build text grid map
    lines.append("\nGrid (row=Y, col=X, step=5):")
    grid = {}
    for s in g.current_level.get_sprites_by_tag("ihdgageizm"):
        grid[(s.x, s.y)] = '#'
    for s in g.current_level.get_sprites_by_tag("gbvqrjtaqo"):
        grid[(s.x, s.y)] = 'B'
    for s in g.current_level.get_sprites_by_tag("npxgalaybz"):
        grid[(s.x, s.y)] = '*'
    for s in g.current_level.get_sprites_by_tag("ttfwljgohq"):
        grid[(s.x, s.y)] = 'S'
    for s in g.current_level.get_sprites_by_tag("soyhouuebz"):
        grid[(s.x, s.y)] = 'C'
    for s in g.current_level.get_sprites_by_tag("rhsxkxzdjz"):
        grid[(s.x, s.y)] = 'R'
    for i, s in enumerate(g.plrpelhym):
        grid[(s.x, s.y)] = f'G'
    grid[(g.gudziatsk.x, g.gudziatsk.y)] = 'P'

    x_positions = sorted(set(x for x, y in grid.keys()))
    y_positions = sorted(set(y for x, y in grid.keys()))

    header = "     " + " ".join(f"{x:3d}" for x in x_positions)
    lines.append(header)
    for y in y_positions:
        row = f"{y:3d}  "
        for x in x_positions:
            cell = grid.get((x, y), ' ')
            row += f"  {cell} "
        lines.append(row)

    return "\n".join(lines)


# ============================================================
# Game Runner
# ============================================================
class GameRunner:
    """Runs the game, solves each level, records gameplay."""

    def __init__(self, game_dir: str, game_id: str = None):
        self.game_dir = game_dir
        self.game = load_game(game_dir)
        self.game_id = game_id or Path(game_dir).name
        self.source_code = Path(os.path.join(game_dir, "source.py")).read_text()
        self.session_id = str(uuid.uuid4())
        self.recording_lines: List[str] = []
        self.level_stats: List[Dict] = []

        # Load metadata
        meta_path = os.path.join(game_dir, "metadata.json")
        self.metadata = json.loads(Path(meta_path).read_text()) if os.path.exists(meta_path) else {}

    def record_step(self, frame_data, action_name: str = "RESET"):
        """Record one game step to JSONL."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "game_id": frame_data.game_id,
                "state": frame_data.state,
                "levels_completed": frame_data.levels_completed,
                "win_levels": frame_data.win_levels,
                "action_input": {
                    "id": action_name,
                    "data": {"game_id": frame_data.game_id},
                    "reasoning": None,
                },
                "guid": frame_data.guid or self.session_id,
                "full_reset": frame_data.full_reset,
                "available_actions": frame_data.available_actions,
                "frame": frame_data.frame,
            }
        }
        self.recording_lines.append(json.dumps(entry))

    def run(self) -> Dict:
        """Run the game, solving each level."""
        from arcengine import ActionInput, GameAction

        print(f"Game: {self.game_id}")
        print(f"Levels: {len(self.game._levels)}")
        print(f"Player size: {self.game.gisrhqpee}x{self.game.tbwnoxqgc}")

        # Initial reset
        reset_frame = self.game.perform_action(ActionInput(id=GameAction.RESET))
        self.record_step(reset_frame, "RESET")

        total_actions = 0
        levels_solved = 0
        game_action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                           3: GameAction.ACTION3, 4: GameAction.ACTION4}

        for level_idx in range(len(self.game._levels)):
            print(f"\n{'='*60}")
            print(f"LEVEL {level_idx + 1}")
            level_start = time.time()
            level_actions = 0
            level_solved = False
            max_retries = 3  # one per life

            for retry in range(max_retries):
                # Build solver from current game state
                solver = GridSolver(self.game)
                actions = solver.solve_level()

                solver_type = "BFS"
                if actions is None:
                    solver_type = "LLM"
                    print(f"  BFS solver failed (retry {retry}), trying LLM...")
                    state_desc = build_state_description(self.game)
                    llm = LLMSolver(self.source_code)
                    actions = llm.solve(state_desc)

                if actions is None:
                    print(f"  No solution found for level {level_idx + 1}")
                    break

                # Print plan info
                budget_info = f"budget={solver.step_budget}" if solver_type == "BFS" else ""
                print(f"  [{solver_type}] Plan: {len(actions)} actions {budget_info}")
                if len(actions) <= 30:
                    print(f"    Actions: {[ACTION_NAMES[a] for a in actions]}")
                else:
                    print(f"    First 20: {[ACTION_NAMES[a] for a in actions[:20]]}...")

                # Execute actions
                prev_level = self.game._current_level_index
                prev_lives = self.game.aqygnziho if hasattr(self.game, 'aqygnziho') else 3

                for i, action in enumerate(actions):
                    ga = game_action_map.get(action)
                    if ga is None:
                        continue

                    frame = self.game.perform_action(ActionInput(id=ga))
                    self.record_step(frame, f"ACTION{action}")
                    level_actions += 1
                    total_actions += 1

                    # Check level advancement
                    if self.game._current_level_index != prev_level:
                        level_solved = True
                        break

                    # Check game state
                    if frame.state == "WIN":
                        level_solved = True
                        break
                    if frame.state == "GAME_OVER":
                        print(f"  GAME OVER at action {i+1}")
                        break

                    # Detect life loss (step exhaustion reset)
                    cur_lives = self.game.aqygnziho if hasattr(self.game, 'aqygnziho') else 3
                    if cur_lives < prev_lives:
                        print(f"  Lost a life (steps exhausted) at action {i+1}, {cur_lives} lives left")
                        prev_lives = cur_lives
                        break  # break out to retry with fresh state

                if level_solved:
                    break
                if frame.state == "GAME_OVER":
                    break

            elapsed = time.time() - level_start
            status = "SOLVED" if level_solved else "FAILED"
            print(f"  >> Level {level_idx + 1}: {status} in {level_actions} actions ({elapsed:.2f}s)")

            self.level_stats.append({
                "level": level_idx + 1,
                "solved": level_solved,
                "actions": level_actions,
                "time_seconds": round(elapsed, 2),
            })

            if level_solved:
                levels_solved += 1
            else:
                break  # stop if level not solved

            if frame.state in ("WIN", "GAME_OVER"):
                break

        return {
            "game_id": self.game_id,
            "levels_solved": levels_solved,
            "total_levels": len(self.game._levels),
            "total_actions": total_actions,
            "level_stats": self.level_stats,
        }

    def save_recording(self, recordings_dir: str) -> str:
        """Save recording to JSONL file."""
        session_dir = os.path.join(recordings_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        game_full_id = self.metadata.get("game_id", self.game_id)
        filename = f"{game_full_id}-{self.session_id}.jsonl"
        filepath = os.path.join(session_dir, filename)
        with open(filepath, "w") as f:
            for line in self.recording_lines:
                f.write(line + "\n")
        return filepath


# ============================================================
# Main
# ============================================================
def main():
    game_id = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    base_dir = Path(__file__).parent
    game_dir = base_dir / "dataset" / "games" / game_id
    recordings_dir = base_dir / "recordings"

    if not game_dir.exists():
        print(f"Error: Game directory not found: {game_dir}")
        sys.exit(1)

    print(f"ARC-AGI-3 LLM Harness")
    print(f"{'='*60}")

    runner = GameRunner(str(game_dir), game_id)
    result = runner.run()

    # Save recording
    rec_path = runner.save_recording(str(recordings_dir))
    print(f"\nRecording: {rec_path}")

    # Save results
    results_path = base_dir / f"{game_id}_results.txt"
    with open(str(results_path), "w") as f:
        f.write(f"ARC-AGI-3 LLM Harness Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Game: {game_id}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Harness: Programmatic BFS solver (from LLM source code analysis)\n")
        f.write(f"         + Claude API fallback for complex levels\n")
        f.write(f"\n")
        f.write(f"RESULTS\n")
        f.write(f"{'-'*60}\n")
        f.write(f"Levels solved: {result['levels_solved']} / {result['total_levels']}\n")
        f.write(f"Total actions: {result['total_actions']}\n\n")

        for stat in result['level_stats']:
            status = "SOLVED" if stat['solved'] else "FAILED"
            f.write(f"  Level {stat['level']}: {status}")
            f.write(f" | {stat['actions']} actions")
            f.write(f" | {stat['time_seconds']}s\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"Session: {runner.session_id}\n")
        f.write(f"Recording: {rec_path}\n")
        f.write(f"\nApproach:\n")
        f.write(f"  The harness uses Claude (LLM) to analyze the game's source.py\n")
        f.write(f"  and understand game mechanics (movement, collision, changers,\n")
        f.write(f"  goals, step budget). From this understanding, it builds a\n")
        f.write(f"  programmatic BFS solver that:\n")
        f.write(f"  1. Extracts obstacle/goal/changer positions from game state\n")
        f.write(f"  2. Computes needed attribute changes (shape/color/rotation)\n")
        f.write(f"  3. Plans optimal waypoint sequence (changers then goal)\n")
        f.write(f"  4. Uses BFS pathfinding between waypoints\n")
        f.write(f"  5. Handles moving changers via time-aware BFS\n")
        f.write(f"  6. Falls back to Claude API for levels that defy BFS\n")

    print(f"Results: {results_path}")
    print(f"\nSummary: {result['levels_solved']}/{result['total_levels']} levels solved, {result['total_actions']} total actions")


if __name__ == "__main__":
    main()
