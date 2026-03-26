"""
Perception module: frame parsing, diffing, state hashing.

Converts raw FrameDataRaw observations into structured state representations
that strategies can consume. Game-agnostic.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Immutable snapshot of a game frame (64x64 grid of color ints 0-15)."""
    grid: tuple  # flattened tuple for hashability
    width: int = 64
    height: int = 64
    mask_rows: tuple = ()  # rows to exclude from state hash (e.g. status bar)

    @classmethod
    def from_raw(cls, frame_data, mask_rows: tuple = ()) -> "Frame":
        """Create from FrameDataRaw.frame (list of grids). Uses last frame."""
        if frame_data is None or frame_data.frame is None:
            return cls(grid=(), width=0, height=0, mask_rows=mask_rows)
        # frame_data.frame is List[List[List[int]]] — take the last animation frame
        raw_grid = frame_data.frame[-1] if frame_data.frame else []
        h = len(raw_grid)
        w = len(raw_grid[0]) if h > 0 else 0
        flat = []
        for row in raw_grid:
            flat.extend(row)
        return cls(grid=tuple(flat), width=w, height=h, mask_rows=mask_rows)

    def to_numpy(self) -> np.ndarray:
        """Return as (H, W) numpy array."""
        if not self.grid:
            return np.zeros((0, 0), dtype=np.int8)
        return np.array(self.grid, dtype=np.int8).reshape(self.height, self.width)

    @property
    def hash(self) -> str:
        """Content hash excluding masked rows (status bar, step counter)."""
        if not self.mask_rows:
            return hashlib.md5(bytes(self.grid)).hexdigest()
        arr = self.to_numpy()
        masked = []
        for y in range(self.height):
            if y not in self.mask_rows:
                masked.extend(arr[y].tolist())
        return hashlib.md5(bytes(masked)).hexdigest()

    @property
    def full_hash(self) -> str:
        """Hash of the complete frame including status bar."""
        return hashlib.md5(bytes(self.grid)).hexdigest()

    def diff(self, other: "Frame") -> "FrameDiff":
        """Compute pixel-level diff between two frames."""
        if self.grid == other.grid:
            return FrameDiff(changed=False, changed_count=0, changed_positions=())
        a = self.to_numpy()
        b = other.to_numpy()
        if a.shape != b.shape:
            return FrameDiff(changed=True, changed_count=-1, changed_positions=())
        mask = a != b
        positions = tuple(zip(*np.where(mask)))
        return FrameDiff(
            changed=True,
            changed_count=int(mask.sum()),
            changed_positions=positions,
        )

    def render_ascii(self, mask_rows: tuple = ()) -> str:
        """Render frame as compact ASCII. Optionally mask bottom rows (status bar)."""
        COLOR_CHARS = "0123456789ABCDEF"
        arr = self.to_numpy()
        lines = []
        for y in range(self.height):
            if y in mask_rows:
                continue
            line = ""
            for x in range(self.width):
                v = arr[y, x]
                line += COLOR_CHARS[v] if 0 <= v < 16 else "?"
            lines.append(line)
        return "\n".join(lines)


@dataclass(frozen=True)
class FrameDiff:
    """Result of comparing two frames."""
    changed: bool
    changed_count: int
    changed_positions: tuple  # ((y, x), ...)


@dataclass
class GameObservation:
    """Structured observation from one game step."""
    frame: Frame
    state_hash: str
    game_state: str  # "IN_PROGRESS", "WIN", "GAME_OVER", "NOT_PLAYED"
    levels_completed: int
    win_levels: int
    available_actions: list
    diff_from_prev: Optional[FrameDiff] = None

    @classmethod
    def from_raw(cls, frame_data, prev_frame: Optional[Frame] = None,
                 mask_rows: tuple = ()) -> "GameObservation":
        """Build observation from FrameDataRaw."""
        frame = Frame.from_raw(frame_data, mask_rows=mask_rows)
        diff = frame.diff(prev_frame) if prev_frame is not None else None

        return cls(
            frame=frame,
            state_hash=frame.hash,
            game_state=frame_data.state.name if frame_data.state else "UNKNOWN",
            levels_completed=getattr(frame_data, "levels_completed", 0),
            win_levels=getattr(frame_data, "win_levels", 0),
            available_actions=list(getattr(frame_data, "available_actions", [])),
            diff_from_prev=diff,
        )
