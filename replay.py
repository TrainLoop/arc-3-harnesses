"""
Replay a recorded game session from a JSONL file.

Usage:
    python replay.py recordings/<scorecard_id>/<game>.jsonl
    python replay.py recordings/<scorecard_id>/<game>.jsonl --fps 10
    python replay.py recordings/<scorecard_id>/<game>.jsonl --mode image --out replay.gif
    python replay.py --latest
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 16-color ARC palette (ANSI 256-color approximations)
ANSI_COLORS = {
    0: 232,   # black
    1: 19,    # blue
    2: 160,   # red
    3: 28,    # green
    4: 226,   # yellow
    5: 243,   # gray
    6: 199,   # magenta
    7: 208,   # orange
    8: 51,    # cyan
    9: 124,   # maroon/dark red
    10: 57,   # indigo
    11: 47,   # lime
    12: 33,   # dark cyan
    13: 89,   # purple
    14: 250,  # light gray
    15: 231,  # white
}


def ansi_pixel(color_id):
    """Render a pixel as two ANSI-colored spaces."""
    c = ANSI_COLORS.get(color_id, 0)
    return f"\033[48;5;{c}m  \033[0m"


def render_frame_terminal(frame, width=64, height=64):
    """Render a 64x64 frame as colored text in the terminal."""
    lines = []
    # Show every other row for better aspect ratio
    for y in range(0, height, 2):
        line = ""
        for x in range(0, width, 2):
            line += ansi_pixel(frame[y][x])
        lines.append(line)
    return "\n".join(lines)


def load_recording(path):
    """Load a JSONL recording file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def replay_terminal(records, fps=5, skip_no_change=True):
    """Play back recording in the terminal."""
    prev_frame = None
    for i, rec in enumerate(records):
        data = rec.get("data", rec)
        frame = data.get("frame")
        if not frame:
            continue

        # Use last frame in sequence (for animations)
        grid = frame[-1] if isinstance(frame[0], list) and isinstance(frame[0][0], list) else frame

        if skip_no_change and prev_frame == grid:
            continue
        prev_frame = grid

        action = data.get("action_input", {}).get("id", "?")
        levels = data.get("levels_completed", 0)
        state = data.get("state", "?")

        # Clear screen and render
        print("\033[2J\033[H", end="")  # clear + home
        print(render_frame_terminal(grid))
        print(f"\nStep {i}/{len(records)} | Action: {action} | "
              f"Levels: {levels}/7 | State: {state}")

        time.sleep(1.0 / fps)


def replay_image(records, out_path="replay.gif", fps=5):
    """Save recording as animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("pip install pillow  (required for image replay)")
        sys.exit(1)

    CSS_PALETTE = [
        (0, 0, 0),       # 0 black
        (0, 116, 217),   # 1 blue
        (255, 65, 54),   # 2 red
        (46, 204, 64),   # 3 green
        (255, 220, 0),   # 4 yellow
        (170, 170, 170), # 5 gray
        (240, 18, 190),  # 6 magenta
        (255, 133, 27),  # 7 orange
        (127, 219, 255), # 8 cyan
        (135, 12, 37),   # 9 maroon
        (55, 33, 128),   # 10 indigo
        (0, 255, 0),     # 11 lime
        (0, 128, 128),   # 12 teal
        (128, 0, 128),   # 13 purple
        (211, 211, 211), # 14 light gray
        (255, 255, 255), # 15 white
    ]

    frames = []
    prev_grid = None
    scale = 4

    for rec in records:
        data = rec.get("data", rec)
        frame = data.get("frame")
        if not frame:
            continue
        grid = frame[-1] if isinstance(frame[0], list) and isinstance(frame[0][0], list) else frame
        if grid == prev_grid:
            continue
        prev_grid = grid

        h, w = len(grid), len(grid[0])
        img = Image.new("RGB", (w * scale, h * scale))
        pixels = img.load()
        for y in range(h):
            for x in range(w):
                color = CSS_PALETTE[grid[y][x] % 16]
                for dy in range(scale):
                    for dx in range(scale):
                        pixels[x * scale + dx, y * scale + dy] = color
        frames.append(img)

    if frames:
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                        duration=int(1000 / fps), loop=0)
        print(f"Saved {len(frames)} frames to {out_path}")
    else:
        print("No frames found in recording")


def find_latest():
    """Find the most recent recording file."""
    rec_dir = Path("recordings")
    if not rec_dir.exists():
        return None
    jsonl_files = sorted(rec_dir.rglob("*.jsonl"), key=os.path.getmtime, reverse=True)
    return str(jsonl_files[0]) if jsonl_files else None


def main():
    parser = argparse.ArgumentParser(description="Replay ARC-AGI-3 game recordings")
    parser.add_argument("file", nargs="?", help="Path to .jsonl recording")
    parser.add_argument("--latest", action="store_true", help="Use most recent recording")
    parser.add_argument("--fps", type=int, default=5, help="Playback speed (default: 5)")
    parser.add_argument("--mode", default="terminal", choices=["terminal", "image"],
                        help="'terminal' for live playback, 'image' for GIF export")
    parser.add_argument("--out", default="replay.gif", help="Output path for image mode")
    args = parser.parse_args()

    path = args.file
    if args.latest or not path:
        path = find_latest()
        if not path:
            print("No recordings found in recordings/")
            sys.exit(1)
        print(f"Using: {path}")

    records = load_recording(path)
    print(f"Loaded {len(records)} steps")

    if args.mode == "terminal":
        replay_terminal(records, fps=args.fps)
    elif args.mode == "image":
        replay_image(records, out_path=args.out, fps=args.fps)


if __name__ == "__main__":
    main()
