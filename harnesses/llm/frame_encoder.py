"""
Encode 64x64 game frames as compact text for LLM consumption.
"""

import numpy as np
from ..perception import Frame, FrameDiff

COLOR_NAMES = {
    0: "K", 1: "B", 2: "R", 3: "G", 4: "Y", 5: "W",
    6: "M", 7: "O", 8: "C", 9: "D", 10: "I", 11: "L",
    12: "T", 13: "P", 14: "S", 15: "X",
}


def encode_rle(frame: Frame, mask_rows: tuple = ()) -> str:
    """Run-length encode the frame. Typical output: 200-600 tokens."""
    arr = frame.to_numpy()
    lines = []
    for y in range(frame.height):
        if y in mask_rows:
            continue
        row = arr[y]
        runs = []
        i = 0
        while i < len(row):
            val = row[i]
            count = 1
            while i + count < len(row) and row[i + count] == val:
                count += 1
            ch = COLOR_NAMES.get(int(val), "?")
            if count > 2:
                runs.append(f"{ch}{count}")
            else:
                runs.append(ch * count)
            i += count
        lines.append(f"{y:2d}|{''.join(runs)}")
    return "\n".join(lines)


def encode_downsampled(frame: Frame, factor: int = 4,
                       mask_rows: tuple = ()) -> str:
    """Downsample to 16x16 by majority color in each block."""
    arr = frame.to_numpy()
    h, w = arr.shape
    out_h, out_w = h // factor, w // factor
    lines = []
    for by in range(out_h):
        if by * factor in mask_rows:
            continue
        row = ""
        for bx in range(out_w):
            block = arr[by*factor:(by+1)*factor, bx*factor:(bx+1)*factor]
            vals, counts = np.unique(block, return_counts=True)
            majority = vals[np.argmax(counts)]
            row += COLOR_NAMES.get(int(majority), "?")
        lines.append(f"{by:2d}|{row}")
    return "\n".join(lines)


def encode_diff(diff: FrameDiff, new_frame: Frame) -> str:
    """Describe what changed between frames."""
    if not diff.changed:
        return "No change (action had no effect - likely hit a wall)"
    if diff.changed_count < 0:
        return "Frame size changed (level transition)"
    if diff.changed_count > 200:
        return f"Major change: {diff.changed_count} pixels changed (possible death/reset/level transition)"

    arr = new_frame.to_numpy()
    changes = []
    for y, x in list(diff.changed_positions)[:30]:
        new_val = COLOR_NAMES.get(int(arr[y, x]), "?")
        changes.append(f"({x},{y})={new_val}")

    text = f"{diff.changed_count} pixels changed"
    if changes:
        text += f": {', '.join(changes)}"
    return text
