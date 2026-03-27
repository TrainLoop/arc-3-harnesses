"""
Runner: connects a harness to the ARC-AGI-3 environment and executes it.

Usage:
    python run_harness.py --game ls20 --strategy llm_agent --online
    python run_harness.py --all --strategy llm_agent --online
    python run_harness.py --game ls20 --strategy random
"""

import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from arc_agi import Arcade, OperationMode
from harnesses.base import Harness, HarnessConfig
from harnesses.strategies import RandomStrategy, LLMAgentStrategy


STRATEGY_REGISTRY = {
    "llm_agent": LLMAgentStrategy,
    "random": RandomStrategy,
}


def build_strategy(config: HarnessConfig):
    cls = STRATEGY_REGISTRY.get(config.strategy_name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {config.strategy_name}. "
                         f"Available: {list(STRATEGY_REGISTRY.keys())}")
    # Merge params, avoiding duplicates
    params = dict(config.strategy_params)
    params.setdefault("game_id", config.game_id)
    params["action_space"] = config.action_space
    return cls(**params)


def get_game_ids(dataset_dir: Path) -> list[str]:
    """Get all game IDs from the dataset."""
    games_dir = dataset_dir / "games"
    if not games_dir.exists():
        return []
    return sorted([d.name for d in games_dir.iterdir()
                   if d.is_dir() and (d / "source.py").exists()])


def run_single_game(config: HarnessConfig, api_key: str, online: bool,
                    render_mode: str = None) -> dict:
    """Run a single game. Returns result summary."""
    print(f"\n{'#'*60}")
    print(f"GAME: {config.game_id}")
    print(f"Strategy: {config.strategy_name}")
    print(f"Action space: {config.action_space}")
    print(f"{'#'*60}")

    op_mode = OperationMode.ONLINE if online else OperationMode.NORMAL
    arc = Arcade(arc_api_key=api_key, operation_mode=op_mode)

    try:
        env = arc.make(config.game_id, save_recording=True, render_mode=render_mode)
    except Exception as e:
        print(f"  ERROR creating environment: {e}")
        return {"game_id": config.game_id, "error": str(e), "levels_completed": 0, "win_levels": 0}

    if env is None:
        print("  ERROR: Failed to create environment")
        return {"game_id": config.game_id, "error": "env is None", "levels_completed": 0, "win_levels": 0}

    # Update action space from the environment
    if hasattr(env, '_last_response') and env._last_response:
        avail = env._last_response.available_actions
        if avail:
            config.action_space = list(avail)
            print(f"  Actions from env: {config.action_space}")

    # Print URLs
    scorecard_id = getattr(arc, '_default_scorecard_id', None)
    guid = getattr(env, '_guid', None)
    if scorecard_id:
        print(f"  Scorecard: https://arcprize.org/scorecards/{scorecard_id}")

    strategy = build_strategy(config)
    harness = Harness(config, strategy)

    print("  Running...")
    print("  " + "-" * 40)
    result = harness.run(env)
    print("  " + "-" * 40)

    print(f"  Result: {'WIN' if result.won else 'NOT WON'} "
          f"| {result.levels_completed}/{result.win_levels} levels "
          f"| {result.total_actions} actions "
          f"| {result.duration_seconds:.1f}s")

    # Close scorecard
    try:
        scorecard = arc.close_scorecard()
        if scorecard:
            print(f"  Score: {scorecard.score}")
    except Exception:
        pass

    return {
        "game_id": config.game_id,
        "won": result.won,
        "levels_completed": result.levels_completed,
        "win_levels": result.win_levels,
        "total_actions": result.total_actions,
        "duration": result.duration_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 harness")
    parser.add_argument("--game", default=None, help="Game ID (e.g. ls20)")
    parser.add_argument("--all", action="store_true", help="Run on all games")
    parser.add_argument("--strategy", default="llm_agent", help="Strategy name")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--max-actions-per-level", type=int, default=500)
    parser.add_argument("--api-key", default=None, help="ARC API key")
    parser.add_argument("--model", default="gpt-oss:20b", help="LLM model")
    parser.add_argument("--backend", default="ollama", help="LLM backend")
    parser.add_argument("--output", default=None, help="Results JSON path")
    parser.add_argument("--render", default=None, choices=["terminal", "human"])
    parser.add_argument("--online", action="store_true",
                        help="Run ONLINE (API-based, enables web replay)")
    args = parser.parse_args()

    if not args.game and not args.all:
        parser.error("Provide --game <id> or --all")

    api_key = args.api_key or os.environ.get("ARC_API_KEY", "")
    dataset_dir = Path(__file__).parent / "dataset"

    # Determine game list
    if args.all:
        game_ids = get_game_ids(dataset_dir)
    else:
        game_ids = [args.game]

    print(f"ARC-AGI-3 Harness Runner")
    print(f"{'='*60}")
    print(f"Games: {len(game_ids)} — {', '.join(game_ids)}")
    print(f"Strategy: {args.strategy}")
    print(f"Model: {args.model}")
    print(f"Online: {args.online}")
    print()

    all_results = []
    for game_id in game_ids:
        # Load metadata to get full game_id and action space
        meta_path = dataset_dir / "games" / game_id / "metadata.json"
        game_full_id = game_id
        default_actions = [1, 2, 3, 4]
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            game_full_id = meta.get("game_id", game_id)

        config = HarnessConfig(
            game_id=game_full_id,
            strategy_name=args.strategy,
            max_actions=args.max_actions,
            max_actions_per_level=args.max_actions_per_level,
            action_space=default_actions,
            mask_rows=list(range(60, 64)),  # mask status bar rows
            strategy_params={
                "model": args.model,
                "backend": args.backend,
                "dataset_dir": str(dataset_dir),
                "game_id": game_id,  # short id for source lookup
            },
        )

        if args.online:
            config.step_delay = 0.1

        result = run_single_game(config, api_key, args.online, args.render)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_levels = sum(r.get("levels_completed", 0) for r in all_results)
    total_win = sum(r.get("win_levels", 0) for r in all_results)
    for r in all_results:
        status = "WIN" if r.get("won") else f"{r.get('levels_completed',0)}/{r.get('win_levels',0)}"
        err = f" [{r['error'][:40]}]" if r.get("error") else ""
        print(f"  {r['game_id']:20s}  {status:>8s}  {r.get('total_actions',0):5d} actions{err}")
    print(f"\nTotal: {total_levels}/{total_win} levels solved")

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()
