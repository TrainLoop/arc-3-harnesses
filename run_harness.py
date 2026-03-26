"""
Runner: connects a harness to the ARC-AGI-3 environment and executes it.

Usage:
    python run_harness.py --game ls20 [--strategy graph_explorer] [--config harnesses/configs/ls20.json]
    python run_harness.py --game ls20 --strategy random
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).parent))

from arc_agi import Arcade, OperationMode
from harnesses.base import Harness, HarnessConfig
from harnesses.strategies import GraphExplorerStrategy, RandomStrategy, SourceSolverStrategy


STRATEGY_REGISTRY = {
    "graph_explorer": GraphExplorerStrategy,
    "random": RandomStrategy,
    "source_solver": SourceSolverStrategy,
}


def build_strategy(config: HarnessConfig):
    """Instantiate a strategy from config."""
    cls = STRATEGY_REGISTRY.get(config.strategy_name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {config.strategy_name}. "
                         f"Available: {list(STRATEGY_REGISTRY.keys())}")
    return cls(
        action_space=config.action_space,
        **config.strategy_params,
    )


def main():
    parser = argparse.ArgumentParser(description="Run an ARC-AGI-3 harness")
    parser.add_argument("--game", required=True, help="Game ID (e.g. ls20)")
    parser.add_argument("--strategy", default=None, help="Strategy name")
    parser.add_argument("--config", default=None, help="Path to harness config JSON")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--api-key", default=None, help="ARC API key")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    parser.add_argument("--render", default=None, choices=["terminal", "human"],
                        help="Render mode: 'terminal' for ASCII, 'human' for matplotlib")
    parser.add_argument("--online", action="store_true",
                        help="Run in ONLINE mode (actions via API, enables web replay)")
    args = parser.parse_args()

    # Load or build config
    if args.config:
        config = HarnessConfig.load(args.config)
    else:
        config_path = Path(__file__).parent / "harnesses" / "configs" / f"{args.game}.json"
        if config_path.exists():
            config = HarnessConfig.load(str(config_path))
            print(f"Loaded config from {config_path}")
        else:
            config = HarnessConfig(
                game_id=args.game,
                strategy_name=args.strategy or "graph_explorer",
            )

    # Override from CLI
    if args.strategy:
        config.strategy_name = args.strategy
    if args.max_actions:
        config.max_actions = args.max_actions
    config.game_id = args.game
    if args.online:
        config.step_delay = 0.15  # 150ms between API calls to avoid rate limits

    # API key
    api_key = args.api_key or os.environ.get("ARC_API_KEY", "")

    print(f"Game: {config.game_id}")
    print(f"Strategy: {config.strategy_name}")
    print(f"Max actions: {config.max_actions}")
    print(f"Action space: {config.action_space}")
    print()

    # Initialize Arcade
    op_mode = OperationMode.ONLINE if args.online else OperationMode.NORMAL
    arc = Arcade(
        arc_api_key=api_key,
        operation_mode=op_mode,
    )

    # Create environment
    print(f"Creating environment for {config.game_id}...")
    env = arc.make(config.game_id, save_recording=True, render_mode=args.render)
    if env is None:
        print("ERROR: Failed to create environment")
        sys.exit(1)

    print(f"Environment ready. Win levels: {env.info.title}")

    # Print viewer URLs early so user can open them while the run is in progress
    scorecard_id = getattr(arc, '_default_scorecard_id', None)
    guid = getattr(env, '_guid', None)
    if scorecard_id:
        print(f"\nScorecard: https://arcprize.org/scorecards/{scorecard_id}")
    if guid:
        print(f"Replay:    https://arcprize.org/replay/{guid}")
    print()

    # Build and run harness
    strategy = build_strategy(config)
    harness = Harness(config, strategy)

    print("Running harness...")
    print("-" * 50)
    result = harness.run(env)
    print("-" * 50)
    print()

    # Results
    print(f"Result: {'WIN' if result.won else 'NOT WON'}")
    print(f"Levels: {result.levels_completed}/{result.win_levels}")
    print(f"Actions: {result.total_actions}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    if hasattr(strategy, 'serialize'):
        stats = strategy.serialize()
        if 'graph_stats' in stats:
            print(f"Graph: {stats['graph_stats']}")

    # Save results
    if args.output:
        Path(args.output).write_text(result.to_json())
        print(f"\nResults saved to {args.output}")

    # Close scorecard
    scorecard = arc.close_scorecard()
    if scorecard:
        print(f"Score: {scorecard.score}")

    # Reprint URLs at the end for convenience
    if scorecard_id:
        print(f"\nScorecard: https://arcprize.org/scorecards/{scorecard_id}")
    if guid:
        print(f"Replay:    https://arcprize.org/replay/{guid}")


if __name__ == "__main__":
    main()
