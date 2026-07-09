"""
BJudge Pipeline — CLI Entry Point

Run with: python -m bjudge

Usage:
    # Live mode (executes real PyAutoGUI actions):
    python -m bjudge \\
        --objective "Open Settings and enable Dark Mode" \\
        --actions actions.json \\
        --mode live \\
        --output results.json

    # Replay mode (processes pre-recorded screenshots):
    python -m bjudge \\
        --objective "Open Settings and enable Dark Mode" \\
        --replay replay_data.json \\
        --mode replay \\
        --output results.json
"""

import argparse
import json
import sys

from bjudge.config import (
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_MAX_TOKENS,
    VISUAL_OUTPUTS_DIR,
    OPENROUTER_API_KEY,
)
from bjudge.pipeline.orchestrator import run_pipeline, save_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BJudge — Test-Time Scaling Pipeline for Computer-Use Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 2 live rollouts from an actions JSON file:
  python -m bjudge \\
      --objective "Open Settings and enable Dark Mode" \\
      --actions data/actions.json \\
      --mode live

  # Replay pre-recorded rollouts:
  python -m bjudge \\
      --objective "Submit the contact form" \\
      --replay data/replay.json \\
      --mode replay \\
      --output data/results.json
        """,
    )

    parser.add_argument(
        "--objective",
        type=str,
        required=True,
        help="Natural language description of the task the agent should accomplish.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "replay"],
        default="replay",
        help="Execution mode: 'live' for real desktop actions, 'replay' for saved screenshots. (default: replay)",
    )

    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing action sequences for LIVE mode. "
            'Format: [["pyautogui.click(x,y)", ...], [...]]'
        ),
    )

    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing replay data for REPLAY mode. "
            "Format: list of rollouts, each being a list of step dicts."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/results.json",
        help="Path to save the pipeline results JSON. (default: data/results.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=VISUAL_OUTPUTS_DIR,
        help="Directory for visual engine outputs. (default: data/visual_outputs/)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=ORCHESTRATOR_MODEL,
        help=f"VLM model slug for narrative generation and evaluation. (default: {ORCHESTRATOR_MODEL})",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=ORCHESTRATOR_MAX_TOKENS,
        help=f"Maximum tokens per VLM call. (default: {ORCHESTRATOR_MAX_TOKENS})",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    if not OPENROUTER_API_KEY:
        print(
            "ERROR: OPENROUTER_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        print(
            "Export it before running: export OPENROUTER_API_KEY='your-key-here'",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "live" and not args.actions:
        print(
            "ERROR: --actions is required for LIVE mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "replay" and not args.replay:
        print(
            "ERROR: --replay is required for REPLAY mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Load Input Data
    # ---------------------------------------------------------------------------

    action_sequences = None
    replay_data = None

    if args.actions:
        with open(args.actions, "r", encoding="utf-8") as f:
            action_sequences = json.load(f)
        print(f"Loaded {len(action_sequences)} action sequence(s) from {args.actions}")

    if args.replay:
        with open(args.replay, "r", encoding="utf-8") as f:
            replay_data = json.load(f)
        print(f"Loaded {len(replay_data)} replay rollout(s) from {args.replay}")

    # ---------------------------------------------------------------------------
    # Run Pipeline
    # ---------------------------------------------------------------------------

    result = run_pipeline(
        objective=args.objective,
        action_sequences=action_sequences,
        replay_data=replay_data,
        mode=args.mode,
        output_dir=args.output_dir,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    # ---------------------------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------------------------

    output_path = save_results(result, args.output)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Objective: {result.objective}")
    print(f"Rollouts:  {len(result.rollouts)}")
    print(f"Winner:    Rollout #{result.winner_index}")
    print(f"{'='*60}")

    if result.evaluator_thoughts:
        print(f"\nEvaluator Reasoning:")
        print(f"  {result.evaluator_thoughts[:500]}")

    best = result.best_rollout
    print(f"\nWinning Narrative:")
    print(f"  {best.narrative[:500]}")


if __name__ == "__main__":
    main()
