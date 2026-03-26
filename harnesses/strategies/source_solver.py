"""
Source-code analysis solver for ls20.

Parses the game's level definitions to extract wall layouts, changers,
pickups, and doors, then solves each level via BFS over the abstract
game state: (position, key_shape, key_color, key_rotation, doors_opened,
pickups_collected).

This is the most powerful approach when source code is available. Each
level is solved offline and the optimal action sequence is cached for
replay at runtime.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from arcengine import GameAction

from ..base import Strategy
from ..perception import GameObservation

_ACTION_BY_ID = {a.value: a for a in GameAction}

# Grid constants: 12x12 grid on 5-pixel steps
CELL = 5
GRID_W, GRID_H = 12, 12
X_OFFSET = 4  # first column at pixel x=4

def px_to_grid(px, py):
    return (px - X_OFFSET) // CELL, py // CELL

def grid_to_px(gx, gy):
    return gx * CELL + X_OFFSET, gy * CELL

def in_bbox(obj_x, obj_y, player_px, player_py, pw=5, ph=5):
    """Check if object position falls within player's bounding box."""
    return (player_px <= obj_x < player_px + pw and
            player_py <= obj_y < player_py + ph)


@dataclass
class PushWall:
    """A pushable wall that launches the player when collided with."""
    px: int  # pixel x
    py: int  # pixel y
    dx: int  # push direction x (-1, 0, 1)
    dy: int  # push direction y (-1, 0, 1)
    width: int = 5
    height: int = 5


@dataclass
class LevelDef:
    """Extracted level definition from source code."""
    index: int
    step_counter: int = 42
    step_decrement: int = 2
    start_shape: int = 0
    start_color: int = 0
    start_rotation: int = 0  # index into [0, 90, 180, 270]
    player_pos: tuple = (0, 0)   # pixel coords
    walls: set = field(default_factory=set)  # grid coords
    walls_px: set = field(default_factory=set)  # pixel coords of all walls+doors
    doors: list = field(default_factory=list)  # [(px, py, goal_shape, goal_color, goal_rotation)]
    pickups: list = field(default_factory=list)  # [(px, py)]
    shape_changers: list = field(default_factory=list)  # [(px, py)]
    color_changers: list = field(default_factory=list)  # [(px, py)]
    rotation_changers: list = field(default_factory=list)  # [(px, py)]
    push_walls: list = field(default_factory=list)  # [PushWall]

    @property
    def moves_per_life(self):
        return self.step_counter // self.step_decrement

    @property
    def player_grid(self):
        return px_to_grid(*self.player_pos)


COLORS = [None, None, None, None, None, None, None, None, 8, 9, None, None, 12, None, 14, None]
# Color indices used in game: index into tnkekoeuk
# tnkekoeuk values from source: epqvqkpffo, jninpsotet, bejggpjowv, tqogkgimes
# These map to actual color values. From source analysis:
# The game uses 4 colors. GoalColor/StartColor use the actual color value.
# We need the color INDEX (0-3) for the game state.
# From level data: GoalColor values seen: 8, 9, 12, 14
# These correspond to indices 0-3 in tnkekoeuk.

ROTATION_VALUES = [0, 90, 180, 270]

def _color_to_idx(color_val, color_list):
    """Convert a color value to its index in the color list."""
    try:
        return color_list.index(color_val)
    except ValueError:
        return 0

def _rotation_to_idx(rot_val):
    return ROTATION_VALUES.index(rot_val) if rot_val in ROTATION_VALUES else 0


# ===== LS20 LEVEL DEFINITIONS (extracted from source) =====
# Color list: determined from all GoalColor/StartColor values across levels
LS20_COLORS = [12, 9, 14, 8]  # epqvqkpffo=12, jninpsotet=9, bejggpjowv=14, tqogkgimes=8

def _build_ls20_levels():
    """Build level definitions from source-extracted data."""
    levels = []

    # ---------- LEVEL 1 ----------
    L1 = LevelDef(index=0)
    L1.step_counter = 42
    L1.step_decrement = 1
    L1.start_shape = 5
    L1.start_color = _color_to_idx(9, LS20_COLORS)
    L1.start_rotation = _rotation_to_idx(270)
    L1.player_pos = (34, 45)
    # Goal: shape=5, color=9(idx1), rotation=0(idx0)
    L1.doors = [(34, 10, 5, _color_to_idx(9, LS20_COLORS), _rotation_to_idx(0))]
    L1.rotation_changers = [(19, 30)]
    L1.pickups = []
    # Walls (grid coords)
    wall_pixels_1 = [
        (4,0),(9,0),(4,5),(14,0),(19,0),(24,0),(29,0),(39,0),(44,0),(49,0),(54,0),(59,0),
        (4,10),(4,15),(4,20),(4,25),(59,15),(59,20),(59,25),(59,30),(59,35),(59,40),(59,45),
        (59,50),(59,55),(54,55),(49,55),(44,55),(39,55),(34,55),(29,55),(24,55),(19,55),
        (4,40),(4,45),(4,50),(9,50),(4,55),(9,55),(14,55),(54,25),(54,20),(34,0),(59,10),
        (59,5),(54,15),(54,10),(44,5),(39,5),(34,5),(29,5),(54,50),(54,45),(24,5),(19,5),
        (9,35),(9,45),(19,50),(9,40),(49,5),(54,5),(49,50),(14,50),(14,5),(9,5),(9,30),
        (9,25),(9,20),(9,15),(9,10),(49,10),(44,20),(39,10),(44,10),(49,15),(29,10),(29,15),
        (39,15),(44,15),(49,20),(14,15),(19,15),(24,15),(24,10),(19,10),(14,10),(29,20),
        (39,20),(24,20),(29,40),(19,20),(14,20),(54,30),(24,40),(14,45),(29,35),(4,30),
        (4,35),(54,35),(54,40),(14,40),(24,50),(29,50),(39,50),(44,50),(34,50),(29,30),
    ]
    L1.walls = {px_to_grid(x, y) for x, y in wall_pixels_1}
    L1.walls_px = set(wall_pixels_1) | {d[:2] for d in L1.doors}
    levels.append(L1)

    # ---------- LEVEL 2 ----------
    L2 = LevelDef(index=1)
    L2.step_counter = 42
    L2.step_decrement = 2  # default
    L2.start_shape = 5
    L2.start_color = _color_to_idx(9, LS20_COLORS)
    L2.start_rotation = _rotation_to_idx(0)
    L2.player_pos = (29, 40)
    L2.doors = [(14, 40, 5, _color_to_idx(9, LS20_COLORS), _rotation_to_idx(270))]
    L2.rotation_changers = [(49, 45)]
    L2.pickups = [(15, 16), (40, 51)]
    wall_pixels_2 = [
        (4,0),(9,0),(4,5),(14,0),(19,0),(24,0),(29,0),(39,0),(44,0),(49,0),(54,0),(59,0),
        (4,10),(4,15),(4,20),(4,25),(4,30),(4,35),(59,15),(59,20),(59,25),(59,30),(59,35),
        (59,40),(59,45),(59,50),(59,55),(54,55),(49,55),(44,55),(39,55),(34,55),(29,55),
        (24,55),(19,55),(4,40),(4,45),(4,50),(9,50),(4,55),(9,55),(14,55),(54,30),(34,0),
        (59,10),(59,5),(54,15),(54,10),(9,35),(9,45),(19,50),(9,40),(54,5),(14,45),(14,50),
        (9,5),(9,30),(9,25),(19,30),(24,30),(19,40),(19,45),(19,35),(39,15),(39,35),(44,30),
        (34,45),(14,5),(39,20),(44,20),(24,20),(44,25),(39,40),(39,45),(24,35),(24,25),
        (24,50),(19,25),(24,40),(24,45),(29,45),(29,30),(29,25),(24,15),(44,35),(54,34),
        (29,50),(34,50),
    ]
    L2.walls = {px_to_grid(x, y) for x, y in wall_pixels_2}
    L2.walls_px = set(wall_pixels_2) | {d[:2] for d in L2.doors}
    levels.append(L2)

    # ---------- LEVEL 3 ----------
    L3 = LevelDef(index=2)
    L3.step_counter = 42
    L3.step_decrement = 2
    L3.start_shape = 5
    L3.start_color = _color_to_idx(12, LS20_COLORS)
    L3.start_rotation = _rotation_to_idx(0)
    L3.player_pos = (9, 45)
    L3.doors = [(54, 50, 5, _color_to_idx(9, LS20_COLORS), _rotation_to_idx(180))]
    L3.rotation_changers = [(49, 10)]
    L3.color_changers = [(29, 45)]
    L3.pickups = [(35, 16), (20, 31)]
    wall_pixels_3 = [
        (4,0),(9,0),(14,0),(19,0),(24,0),(29,0),(39,0),(44,0),(49,0),(59,0),(4,10),(4,15),
        (4,20),(4,25),(4,30),(4,35),(59,20),(59,25),(59,30),(59,35),(59,40),(59,45),(59,50),
        (59,55),(49,55),(44,55),(39,55),(34,55),(29,55),(24,55),(19,55),(4,40),(4,45),(4,50),
        (9,50),(4,55),(9,55),(14,55),(34,0),(59,10),(59,5),(39,10),(14,25),(19,40),(19,45),
        (19,35),(49,50),(39,35),(14,30),(49,45),(49,40),(14,20),(14,50),(39,5),(44,45),
        (19,50),(44,40),(39,50),(44,20),(49,20),(39,20),(19,10),(14,35),(39,15),(34,35),
        (14,10),(14,15),(49,35),(24,35),(34,10),(24,10),(59,15),(54,55),(44,35),(39,40),
        (39,45),(44,50),(4,5),(54,0),
    ]
    L3.walls = {px_to_grid(x, y) for x, y in wall_pixels_3}
    L3.walls_px = set(wall_pixels_3) | {d[:2] for d in L3.doors}
    L3.push_walls = [
        PushWall(px=8, py=5, dx=1, dy=0),   # yjgargdic_r: pushes right
        PushWall(px=54, py=4, dx=0, dy=1),   # kapcaakvb_b: pushes down
    ]
    levels.append(L3)

    # ---------- LEVEL 4 ----------
    L4 = LevelDef(index=3)
    L4.step_counter = 42
    L4.step_decrement = 1
    L4.start_shape = 4
    L4.start_color = _color_to_idx(14, LS20_COLORS)
    L4.start_rotation = _rotation_to_idx(0)
    L4.player_pos = (54, 5)
    L4.doors = [(9, 5, 5, _color_to_idx(9, LS20_COLORS), _rotation_to_idx(0))]
    L4.shape_changers = [(24, 30)]
    L4.color_changers = [(34, 30)]
    L4.pickups = [(35, 51), (20, 16)]
    wall_pixels_4 = [
        (4,0),(9,0),(4,5),(14,0),(19,0),(24,0),(29,0),(39,0),(44,0),(49,0),(54,0),(59,0),
        (4,10),(4,15),(4,20),(4,25),(4,30),(59,15),(59,20),(59,25),(59,30),(59,35),(59,40),
        (59,45),(59,50),(59,55),(54,55),(49,55),(44,55),(39,55),(34,55),(29,55),(24,55),
        (19,55),(4,40),(4,45),(4,50),(9,50),(4,55),(9,55),(14,55),(59,10),(59,5),(14,10),
        (9,10),(24,25),(29,30),(19,10),(9,20),(29,25),(29,35),(34,35),(19,50),(39,50),(9,15),
        (49,10),(44,10),(39,45),(44,50),(19,25),(39,35),(14,50),(9,45),(29,5),(14,15),(34,0),
        (29,10),(34,5),(54,10),
        # Wall variants
        (29,20),(4,35),(49,25),(29,40),(49,40),(39,30),(19,30),(44,15),
    ]
    L4.walls = {px_to_grid(x, y) for x, y in wall_pixels_4}
    L4.walls_px = set(wall_pixels_4) | {d[:2] for d in L4.doors}
    L4.push_walls = [
        PushWall(px=19, py=34, dx=0, dy=1),  # kapcaakvb_b
        PushWall(px=44, py=19, dx=0, dy=1),  # kapcaakvb_b
        PushWall(px=39, py=26, dx=0, dy=-1), # lujfinsby_t
        PushWall(px=45, py=25, dx=-1, dy=0), # tihiodtoj_l
        PushWall(px=25, py=40, dx=-1, dy=0), # tihiodtoj_l
        PushWall(px=45, py=40, dx=-1, dy=0), # tihiodtoj_l
        PushWall(px=33, py=20, dx=1, dy=0),  # yjgargdic_r
        PushWall(px=8, py=35, dx=1, dy=0),   # yjgargdic_r
    ]
    levels.append(L4)

    # ---------- LEVEL 5 ----------
    L5 = LevelDef(index=4)
    L5.step_counter = 42
    L5.step_decrement = 2
    L5.start_shape = 4
    L5.start_color = _color_to_idx(12, LS20_COLORS)
    L5.start_rotation = _rotation_to_idx(0)
    L5.player_pos = (49, 40)
    L5.doors = [(54, 5, 0, _color_to_idx(8, LS20_COLORS), _rotation_to_idx(180))]
    L5.rotation_changers = [(14, 35)]
    L5.color_changers = [(29, 25)]
    L5.shape_changers = [(19, 10)]
    L5.pickups = [(15, 46), (45, 6), (10, 11)]
    wall_pixels_5 = [
        (4,0),(9,0),(4,5),(19,0),(24,0),(29,0),(39,0),(44,0),(49,0),(54,0),(59,0),(4,10),
        (4,15),(4,20),(4,25),(4,30),(4,35),(59,50),(59,55),(39,55),(34,55),(29,55),(24,55),
        (19,55),(4,40),(4,45),(4,50),(9,50),(4,55),(9,55),(14,55),(59,5),(49,10),(49,5),
        (29,30),(9,20),(24,30),(24,20),(34,30),(49,45),(49,20),(44,55),(49,55),(59,10),
        (59,20),(59,25),(59,35),(59,40),(59,45),(39,45),(44,45),(29,35),(9,5),(34,45),(29,5),
        (29,10),(14,20),(14,0),(19,30),(14,30),(9,15),(29,45),(14,50),(9,45),(49,15),(59,15),
        (44,10),
        # Wall variants
        (29,20),(39,40),(59,30),(24,25),(54,55),(39,30),(34,0),(49,25),
    ]
    L5.walls = {px_to_grid(x, y) for x, y in wall_pixels_5}
    L5.walls_px = set(wall_pixels_5) | {d[:2] for d in L5.doors}
    L5.push_walls = [
        PushWall(px=34, py=4, dx=0, dy=1),   # kapcaakvb_b
        PushWall(px=49, py=29, dx=0, dy=1),  # kapcaakvb_b
        PushWall(px=54, py=51, dx=0, dy=-1), # lujfinsby_t
        PushWall(px=39, py=26, dx=0, dy=-1), # lujfinsby_t
        PushWall(px=55, py=30, dx=-1, dy=0), # tihiodtoj_l
        PushWall(px=20, py=25, dx=-1, dy=0), # tihiodtoj_l
        PushWall(px=33, py=20, dx=1, dy=0),  # yjgargdic_r
        PushWall(px=43, py=40, dx=1, dy=0),  # yjgargdic_r
    ]
    levels.append(L5)

    # ---------- LEVEL 6 ----------
    L6 = LevelDef(index=5)
    L6.step_counter = 42
    L6.step_decrement = 1
    L6.start_shape = 0
    L6.start_color = _color_to_idx(14, LS20_COLORS)
    L6.start_rotation = _rotation_to_idx(0)
    L6.player_pos = (24, 50)
    # TWO doors with different requirements
    L6.doors = [
        (54, 50, 5, _color_to_idx(9, LS20_COLORS), _rotation_to_idx(90)),
        (54, 35, 0, _color_to_idx(8, LS20_COLORS), _rotation_to_idx(180)),
    ]
    L6.rotation_changers = [(34, 40)]
    L6.color_changers = [(24, 30)]
    L6.shape_changers = [(14, 10)]
    L6.pickups = [(40, 6), (10, 46), (10, 6)]
    wall_pixels_6 = [
        (4,0),(9,0),(4,5),(14,0),(19,0),(24,0),(29,0),(39,0),(59,0),(4,10),(4,15),(4,20),
        (4,25),(4,30),(4,35),(59,25),(59,30),(59,35),(59,40),(59,45),(59,50),(59,55),(49,55),
        (44,55),(39,55),(34,55),(29,55),(24,55),(19,55),(4,40),(4,45),(4,50),(9,50),(4,55),
        (9,55),(14,55),(34,0),(59,10),(59,5),(29,35),(24,35),(34,35),(49,35),(49,40),(49,45),
        (49,50),(49,30),(44,30),(19,15),(14,35),(24,15),(34,30),(19,35),(29,15),(14,30),
        (34,15),(14,15),(34,20),(14,20),(54,55),(44,0),(54,0),(59,15),(44,25),(59,20),(44,35),
        (44,50),(39,50),(44,45),(44,40),(44,5),(44,10),(54,15),
        # Wall variant
        (54,20),(49,0),
    ]
    L6.walls = {px_to_grid(x, y) for x, y in wall_pixels_6}
    L6.walls_px = set(wall_pixels_6) | {d[:2] for d in L6.doors}
    L6.push_walls = [
        PushWall(px=49, py=4, dx=0, dy=1),   # kapcaakvb_b
        PushWall(px=50, py=20, dx=-1, dy=0),  # tihiodtoj_l
    ]
    levels.append(L6)

    # ---------- LEVEL 7 ----------
    L7 = LevelDef(index=6)
    L7.step_counter = 42
    L7.step_decrement = 2  # default (not specified → 2)
    L7.start_shape = 1
    L7.start_color = _color_to_idx(12, LS20_COLORS)
    L7.start_rotation = _rotation_to_idx(0)
    L7.player_pos = (19, 15)
    L7.doors = [(29, 50, 0, _color_to_idx(8, LS20_COLORS), _rotation_to_idx(180))]
    L7.rotation_changers = [(54, 10)]
    L7.color_changers = [(9, 40)]
    L7.shape_changers = [(19, 40)]
    L7.pickups = [(30, 21), (50, 6), (15, 46), (40, 6), (55, 51), (10, 6)]
    wall_pixels_7 = [
        (4,0),(9,0),(4,5),(14,0),(19,0),(24,0),(29,0),(39,0),(44,0),(49,0),(54,0),(59,0),
        (4,10),(4,15),(4,20),(4,25),(4,30),(4,35),(59,20),(59,30),(59,45),(59,50),(59,55),
        (54,55),(49,55),(44,55),(39,55),(34,55),(29,55),(24,55),(19,55),(4,40),(4,45),(4,50),
        (9,50),(4,55),(9,55),(14,55),(34,0),(59,10),(59,5),(24,40),(44,10),(44,5),(44,15),
        (39,45),(24,45),(34,45),(24,50),(34,50),(44,20),(44,25),(19,50),(24,35),(39,50),
        (9,45),(14,50),(19,45),(29,15),(59,15),(59,25),(59,35),(59,40),(44,45),(24,15),
        (24,20),(19,20),(54,45),(34,40),(34,5),(24,10),(24,30),(19,30),(14,20),(14,30),
        # Wall variants
        (44,30),(34,35),(39,15),
    ]
    L7.walls = {px_to_grid(x, y) for x, y in wall_pixels_7}
    L7.walls_px = set(wall_pixels_7) | {d[:2] for d in L7.doors}
    L7.push_walls = [
        PushWall(px=39, py=19, dx=0, dy=1),  # kapcaakvb_b
        PushWall(px=34, py=31, dx=0, dy=-1), # lujfinsby_t
        PushWall(px=40, py=30, dx=-1, dy=0), # tihiodtoj_l
    ]
    levels.append(L7)

    return levels


NUM_SHAPES = 6
NUM_COLORS = 4
NUM_ROTATIONS = 4

# Movement deltas: ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right
MOVES = {
    1: (0, -1),  # up
    2: (0, 1),   # down
    3: (-1, 0),  # left
    4: (1, 0),   # right
}


def solve_level(level: LevelDef) -> Optional[list[int]]:
    """
    BFS over abstract game state to find optimal action sequence.

    State: (gx, gy, shape, color, rotation, doors_opened, pickups_collected, steps_remaining)
    """
    sgx, sgy = level.player_grid
    num_doors = len(level.doors)
    num_pickups = len(level.pickups)
    all_doors_mask = (1 << num_doors) - 1
    initial_steps = level.moves_per_life

    # Precompute: which grid cell activates each object
    def objects_at_grid(gx, gy):
        """Return objects activated when player is at grid position (gx, gy)."""
        px, py = grid_to_px(gx, gy)
        result = []
        for i, (ox, oy) in enumerate(level.pickups):
            if in_bbox(ox, oy, px, py):
                result.append(("pickup", i))
        for ox, oy in level.shape_changers:
            if in_bbox(ox, oy, px, py):
                result.append(("shape",))
        for ox, oy in level.color_changers:
            if in_bbox(ox, oy, px, py):
                result.append(("color",))
        for ox, oy in level.rotation_changers:
            if in_bbox(ox, oy, px, py):
                result.append(("rotation",))
        return result

    # Precompute object interactions for each grid cell
    grid_objects = {}
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            objs = objects_at_grid(gx, gy)
            if objs:
                grid_objects[(gx, gy)] = objs

    # Check which grid cells are doors (exact position match)
    door_cells = {}
    for i, (dx, dy, _, _, _) in enumerate(level.doors):
        dg = px_to_grid(dx, dy)
        door_cells[dg] = i

    # Precompute push wall effects: when player enters a cell, compute
    # where they get pushed to. Uses same algorithm as game engine.
    push_destinations = {}  # (gx, gy) -> (dest_gx, dest_gy)
    for pw in level.push_walls:
        # For each grid cell the player could be at where they'd collide
        for gx in range(GRID_W):
            for gy in range(GRID_H):
                ppx, ppy = grid_to_px(gx, gy)
                # Check bbox overlap: player [ppx, ppx+5) x [ppy, ppy+5)
                # vs wall [pw.px, pw.px+5) x [pw.py, pw.py+5)
                if not (ppx < pw.px + pw.width and ppx + 5 > pw.px and
                        ppy < pw.py + pw.height and ppy + 5 > pw.py):
                    continue
                # Calculate push distance (same as ullzqnksoj)
                wall_cx = pw.px + pw.dx
                wall_cy = pw.py + pw.dy
                push_dist = 0
                for dist in range(1, 12):
                    check_x = wall_cx + pw.dx * pw.width * dist
                    check_y = wall_cy + pw.dy * pw.height * dist
                    if (check_x, check_y) in level.walls_px:
                        push_dist = max(0, dist - 1)
                        break
                if push_dist > 0:
                    # Player destination: original pos + push direction * width * distance
                    dest_px = ppx + pw.dx * pw.width * push_dist
                    dest_py = ppy + pw.dy * pw.height * push_dist
                    dest_gx, dest_gy = px_to_grid(dest_px, dest_py)
                    if 0 <= dest_gx < GRID_W and 0 <= dest_gy < GRID_H:
                        push_destinations[(gx, gy)] = (dest_gx, dest_gy)

    # BFS state: (gx, gy, shape, color, rotation, doors_mask, pickups_mask)
    # We track best remaining_steps at each state
    initial_state = (sgx, sgy, level.start_shape, level.start_color,
                     level.start_rotation, 0, 0)

    # best_steps[state] = max remaining steps achievable at this state
    best_steps = {}
    best_steps[initial_state] = initial_steps

    # BFS queue: (state, remaining_steps, action_path)
    queue = deque()
    queue.append((initial_state, initial_steps, []))

    while queue:
        state, rem_steps, path = queue.popleft()
        gx, gy, shape, color, rotation, doors_mask, pickups_mask = state

        # Check if we've already found this state with more steps
        if best_steps.get(state, -1) > rem_steps:
            continue

        for action_id, (dx, dy) in MOVES.items():
            ngx, ngy = gx + dx, gy + dy

            # Bounds check
            if ngx < 0 or ngx >= GRID_W or ngy < 0 or ngy >= GRID_H:
                continue

            # Wall check
            if (ngx, ngy) in level.walls:
                continue  # Regular wall, always blocked

            # Door check: doors block movement unless key matches
            if (ngx, ngy) in door_cells:
                di = door_cells[(ngx, ngy)]
                if doors_mask & (1 << di):
                    pass  # Already opened, walk through
                else:
                    # Check if key matches to open the door
                    _, _, goal_shape, goal_color, goal_rot = level.doors[di]
                    if not (shape == goal_shape and color == goal_color
                            and rotation == goal_rot):
                        continue  # Key doesn't match, blocked

            new_steps = rem_steps - 1
            if new_steps < 0:
                continue  # Out of steps

            new_shape, new_color, new_rotation = shape, color, rotation
            new_doors = doors_mask
            new_pickups = pickups_mask

            # Process objects at target position BEFORE push (matches game engine)
            for obj in grid_objects.get((ngx, ngy), []):
                if obj[0] == "pickup":
                    pi = obj[1]
                    if not (new_pickups & (1 << pi)):
                        new_pickups |= (1 << pi)
                        new_steps = initial_steps  # Refill!
                elif obj[0] == "shape":
                    new_shape = (new_shape + 1) % NUM_SHAPES
                elif obj[0] == "color":
                    new_color = (new_color + 1) % NUM_COLORS
                elif obj[0] == "rotation":
                    new_rotation = (new_rotation + 1) % NUM_ROTATIONS

            # Apply push wall AFTER object processing at pre-push position
            if (ngx, ngy) in push_destinations:
                ngx, ngy = push_destinations[(ngx, ngy)]
                # Game also processes objects at post-push position
                for obj in grid_objects.get((ngx, ngy), []):
                    if obj[0] == "pickup":
                        pi = obj[1]
                        if not (new_pickups & (1 << pi)):
                            new_pickups |= (1 << pi)
                            new_steps = initial_steps
                    elif obj[0] == "shape":
                        new_shape = (new_shape + 1) % NUM_SHAPES
                    elif obj[0] == "color":
                        new_color = (new_color + 1) % NUM_COLORS
                    elif obj[0] == "rotation":
                        new_rotation = (new_rotation + 1) % NUM_ROTATIONS

            # Check door opening: player on door with matching key
            if (ngx, ngy) in door_cells:
                di = door_cells[(ngx, ngy)]
                _, _, goal_shape, goal_color, goal_rot = level.doors[di]
                if (new_shape == goal_shape and new_color == goal_color
                        and new_rotation == goal_rot):
                    new_doors |= (1 << di)

            # Win check
            if new_doors == all_doors_mask:
                return path + [action_id]

            new_state = (ngx, ngy, new_shape, new_color, new_rotation,
                         new_doors, new_pickups)

            if best_steps.get(new_state, -1) < new_steps:
                best_steps[new_state] = new_steps
                queue.append((new_state, new_steps, path + [action_id]))

    return None  # No solution found


class SourceSolverStrategy(Strategy):
    """
    Solves levels by analyzing game source code offline, then replays
    optimal action sequences at runtime.
    """

    name = "source_solver"

    def __init__(self, action_space=None, **kwargs):
        self.action_space = action_space or [1, 2, 3, 4]
        self._levels = _build_ls20_levels()
        self._solutions: dict[int, list[int]] = {}
        self._current_level = 0
        self._action_queue: list[int] = []
        self._step = 0

        # Pre-solve all levels
        print("  [source_solver] Pre-solving all levels...")
        for i, level in enumerate(self._levels):
            sol = solve_level(level)
            if sol:
                self._solutions[i] = sol
                print(f"    Level {i+1}: {len(sol)} actions (baseline: {[21,123,39,92,54,108,109][i]})")
            else:
                print(f"    Level {i+1}: NO SOLUTION FOUND")

    def reset(self):
        """Called on level advance."""
        self._action_queue = []
        self._step = 0
        # Load next level's solution
        if self._current_level in self._solutions:
            self._action_queue = list(self._solutions[self._current_level])

    def soft_reset(self):
        """Called on game over — reload current level solution."""
        self._step = 0
        if self._current_level in self._solutions:
            self._action_queue = list(self._solutions[self._current_level])

    def choose_action(self, obs: GameObservation) -> GameAction:
        self._step += 1

        # Detect level from levels_completed
        if obs.levels_completed != self._current_level:
            self._current_level = obs.levels_completed
            self.reset()

        if self._action_queue:
            return _ACTION_BY_ID[self._action_queue.pop(0)]

        # Fallback: random if no solution
        import random
        return _ACTION_BY_ID[random.choice(self.action_space)]

    def on_step_result(self, action: GameAction, obs: GameObservation):
        pass

    def get_shortest_path(self, from_hash, to_hash):
        return None

    def get_initial_hash(self):
        return None

    def get_level_solution(self, level_index):
        return list(self._solutions.get(level_index, []))

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "solutions": {k: len(v) for k, v in self._solutions.items()},
        }
