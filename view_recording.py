#!/usr/bin/env python3
"""Simple terminal viewer for ARC-AGI-3 game recordings."""

import json
import sys
import os
import time

# ARC color palette (approximate terminal colors)
ARC_COLORS = {
    0: "\033[97m",   # white
    1: "\033[34m",   # blue
    2: "\033[31m",   # red
    3: "\033[32m",   # green
    4: "\033[33m",   # yellow
    5: "\033[90m",   # gray
    6: "\033[35m",   # magenta
    7: "\033[91m",   # orange-ish
    8: "\033[36m",   # cyan
    9: "\033[94m",   # light blue
    10: "\033[95m",  # pink
    11: "\033[93m",  # light yellow
    12: "\033[96m",  # light cyan
    13: "\033[37m",  # light gray
    14: "\033[92m",  # light green
    15: "\033[30m",  # black
}
RESET = "\033[0m"
BLOCK = "\u2588\u2588"  # full block character (doubled for aspect ratio)


def render_frame(frame, scale=4):
    """Render a frame to terminal using colored blocks."""
    if not frame:
        return "Empty frame"

    h = len(frame)
    w = len(frame[0]) if h > 0 else 0
    lines = []

    for y in range(0, h, scale):
        line = ""
        for x in range(0, w, scale):
            # Sample the pixel at this position
            pixel = frame[y][x]
            if isinstance(pixel, list):
                pixel = pixel[0] if pixel else 0
            color = ARC_COLORS.get(pixel, "\033[90m")
            line += f"{color}{BLOCK}{RESET}"
        lines.append(line)

    return "\n".join(lines)


def main():
    # Find recording file
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Find most recent recording
        rec_dir = os.path.join(os.path.dirname(__file__), "recordings")
        sessions = sorted(os.listdir(rec_dir), key=lambda d: os.path.getmtime(os.path.join(rec_dir, d)), reverse=True)
        for session in sessions:
            session_path = os.path.join(rec_dir, session)
            if os.path.isdir(session_path):
                files = [f for f in os.listdir(session_path) if f.endswith(".jsonl")]
                if files:
                    path = os.path.join(session_path, files[0])
                    break
        else:
            print("No recording found")
            sys.exit(1)

    print(f"Recording: {path}")

    # Load all frames
    frames = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            data = entry["data"]
            frames.append({
                "frame": data.get("frame", []),
                "action": data.get("action_input", {}).get("id", "?"),
                "state": data.get("state", "?"),
                "levels_completed": data.get("levels_completed", 0),
                "win_levels": data.get("win_levels", 0),
            })

    print(f"Frames: {len(frames)}")
    print(f"Controls: Enter=next, 'p'=play all, 'q'=quit, number=jump to frame")
    print()

    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    idx = 0

    while idx < len(frames):
        f = frames[idx]
        # Clear screen
        print("\033[2J\033[H", end="")

        header = (f"Frame {idx}/{len(frames)-1} | "
                  f"Action: {f['action']} | "
                  f"State: {f['state']} | "
                  f"Levels: {f['levels_completed']}/{f['win_levels']}")
        print(header)
        print("-" * len(header))

        if f["frame"]:
            frame_data = f["frame"]
            # Handle nested list (might be [frame] or [[rows]])
            if frame_data and isinstance(frame_data[0], list) and isinstance(frame_data[0][0], list):
                frame_data = frame_data[0]  # unwrap outer list
            print(render_frame(frame_data, scale))
        else:
            print("(no frame data)")

        print(f"\n[Enter=next, p=play, q=quit, number=jump]", end=" ")

        try:
            inp = input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if inp == "q":
            break
        elif inp == "p":
            # Play all remaining frames
            for i in range(idx, len(frames)):
                f = frames[i]
                print("\033[2J\033[H", end="")
                header = (f"Frame {i}/{len(frames)-1} | "
                          f"Action: {f['action']} | "
                          f"Levels: {f['levels_completed']}/{f['win_levels']}")
                print(header)
                if f["frame"]:
                    fd = f["frame"]
                    if fd and isinstance(fd[0], list) and isinstance(fd[0][0], list):
                        fd = fd[0]
                    print(render_frame(fd, scale))
                time.sleep(0.15)
            idx = len(frames)
        elif inp.isdigit():
            idx = min(int(inp), len(frames) - 1)
        else:
            idx += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
