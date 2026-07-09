"""
BJudge Pipeline — Orchestrator

Top-level pipeline controller that ties all modules together:

    For each of N rollouts:
        For each step in the rollout:
            1. visual_engine.annotate_and_crop()    (or capture_and_process for live)
            2. Execute PyAutoGUI action
            3. time.sleep(POST_ACTION_DELAY)         (UI latency mitigation)
            4. Capture "after" screenshot
            5. Visual engine processing
            6. narrative_generator.generate_step_narrative()
        Aggregate step narratives → full trajectory narrative

    Collect N trajectory narratives
    evaluator.evaluate(objective, narratives) → winner_index
    Return PipelineResult

Supports two execution modes:
    - LIVE:   Actually executes actions on the macOS desktop
    - REPLAY: Processes pre-recorded screenshot sequences (for testing)
"""

import json
import os
import time
from typing import List, Optional, Literal, Tuple

from src.config import (
    POST_ACTION_DELAY,
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_MAX_TOKENS,
    VISUAL_OUTPUTS_DIR,
)
from src.trajectory_models import (
    Step,
    Trajectory,
    RolloutResult,
    PipelineResult,
)
from src.visual_engine import (
    capture_screenshot,
    annotate_and_crop,
)
from src.narrative_generator import (
    generate_trajectory_narrative,
    generate_narrative_from_paths,
)
from src.evaluator import evaluate


ExecutionMode = Literal["live", "replay"]


# ---------------------------------------------------------------------------
# Action Parsing
# ---------------------------------------------------------------------------

def _parse_action_type(action_cmd: str) -> Tuple[str, int, int, Optional[Tuple[int, int]]]:
    """
    Parse a PyAutoGUI command string to extract the action type and coordinates.

    Supports:
        pyautogui.click(x, y)
        pyautogui.moveTo(x, y)
        pyautogui.moveTo(x, y, duration=...)
        pyautogui.drag(dx, dy)  (relative)
        pyautogui.hotkey('key1', 'key2')
        pyautogui.typewrite('text')

    Returns:
        Tuple of (action_type, x, y, drag_target_or_none)
    """
    import re

    cmd = action_cmd.strip()

    # Click
    click_match = re.match(r"pyautogui\.click\(\s*(\d+)\s*,\s*(\d+)", cmd)
    if click_match:
        x, y = int(click_match.group(1)), int(click_match.group(2))
        return "click", x, y, None

    # MoveTo
    move_match = re.match(r"pyautogui\.moveTo\(\s*(\d+)\s*,\s*(\d+)", cmd)
    if move_match:
        x, y = int(move_match.group(1)), int(move_match.group(2))
        return "moveto", x, y, None

    # Drag (absolute target)
    drag_match = re.match(
        r"pyautogui\.drag\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", cmd
    )
    if drag_match:
        x1 = int(drag_match.group(1))
        y1 = int(drag_match.group(2))
        x2 = int(drag_match.group(3))
        y2 = int(drag_match.group(4))
        return "dragto", x1, y1, (x2, y2)

    # Keyboard actions — use center of screen as default marker position
    # (these don't have meaningful coordinates)
    return "click", 0, 0, None


def _is_keyboard_action(action_cmd: str) -> bool:
    """Check if an action is keyboard-only (no coordinates)."""
    keyboard_prefixes = ["pyautogui.hotkey", "pyautogui.typewrite", "pyautogui.press"]
    return any(action_cmd.strip().startswith(p) for p in keyboard_prefixes)


# ---------------------------------------------------------------------------
# Live Execution Mode
# ---------------------------------------------------------------------------

def _execute_action(action_cmd: str) -> None:
    """
    Execute a PyAutoGUI action string on the live OS.

    WARNING: This actually performs GUI actions on the current desktop.

    Args:
        action_cmd: A valid PyAutoGUI command string.
    """
    import pyautogui
    # Safety: set a short pause to prevent accidental rapid-fire
    pyautogui.PAUSE = 0.5
    try:
        exec(action_cmd)  # noqa: S102 — Controlled execution of known commands
    except Exception as e:
        print(f"  [Warning] Action execution failed: {action_cmd} → {e}")


def run_live_rollout(
    rollout_id: int,
    action_sequence: List[str],
    output_dir: str,
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
) -> RolloutResult:
    """
    Execute a single live rollout on the macOS desktop.

    For each action in the sequence:
    1. Capture "before" screenshot
    2. Execute the PyAutoGUI action
    3. Wait POST_ACTION_DELAY seconds for UI to settle
    4. Capture "after" screenshot
    5. Run visual engine (annotate + crop)
    6. Generate step narrative via VLM

    Args:
        rollout_id: Integer ID for this rollout.
        action_sequence: Ordered list of PyAutoGUI command strings.
        output_dir: Base directory for visual outputs.
        model: VLM model slug.
        max_tokens: Max tokens per narrative call.

    Returns:
        A RolloutResult with trajectory, narratives, and aggregated text.
    """
    rollout_dir = os.path.join(output_dir, f"rollout_{rollout_id}")
    os.makedirs(rollout_dir, exist_ok=True)

    steps: List[Step] = []

    for i, action_cmd in enumerate(action_sequence):
        step_num = i + 1
        step_id = f"step_{step_num:03d}"
        step_dir = os.path.join(rollout_dir, step_id)
        os.makedirs(step_dir, exist_ok=True)

        print(f"  Rollout {rollout_id} | Step {step_num}: {action_cmd}")

        # 1. Capture BEFORE screenshot
        before_path = os.path.join(step_dir, "before.png")
        capture_screenshot(before_path)

        # 2. Execute the action
        if not _is_keyboard_action(action_cmd):
            _execute_action(action_cmd)
        else:
            _execute_action(action_cmd)

        # 3. Wait for UI to settle (3-second delay)
        time.sleep(POST_ACTION_DELAY)

        # 4. Capture AFTER screenshot
        after_path = os.path.join(step_dir, "after.png")
        capture_screenshot(after_path)

        # 5. Visual engine: annotate + crop
        action_type, lx, ly, drag_target = _parse_action_type(action_cmd)
        annotated_path, crop_path = annotate_and_crop(
            screenshot_path=before_path,
            action_type=action_type,
            logical_x=lx,
            logical_y=ly,
            output_dir=step_dir,
            step_id=step_id,
            drag_target_logical=drag_target,
        )

        step = Step(
            step_number=step_num,
            before_image_path=before_path,
            action_description=action_cmd,
            after_image_path=after_path,
            annotated_image_path=annotated_path,
            crop_image_path=crop_path,
        )
        steps.append(step)

    trajectory = Trajectory(rollout_id=rollout_id, steps=steps)

    # 6. Generate full trajectory narrative via VLM
    print(f"  Rollout {rollout_id} | Generating behavior narrative...")
    rollout_result = generate_trajectory_narrative(
        trajectory=trajectory,
        model=model,
        max_tokens=max_tokens,
    )

    return rollout_result


# ---------------------------------------------------------------------------
# Replay Execution Mode
# ---------------------------------------------------------------------------

def run_replay_rollout(
    rollout_id: int,
    steps_data: List[dict],
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
) -> RolloutResult:
    """
    Process a pre-recorded rollout from saved screenshot files.

    No live OS interaction — purely processes existing image files.

    Args:
        rollout_id: Integer ID for this rollout.
        steps_data: List of step dicts with keys:
            step_number, action_description, before_image_path,
            after_image_path, annotated_image_path, crop_image_path
        model: VLM model slug.
        max_tokens: Max tokens per narrative call.

    Returns:
        A RolloutResult with the trajectory narrative.
    """
    print(f"  Replay Rollout {rollout_id} | {len(steps_data)} steps")
    return generate_narrative_from_paths(
        rollout_id=rollout_id,
        steps_data=steps_data,
        model=model,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    objective: str,
    action_sequences: Optional[List[List[str]]] = None,
    replay_data: Optional[List[List[dict]]] = None,
    mode: ExecutionMode = "replay",
    output_dir: str = VISUAL_OUTPUTS_DIR,
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
) -> PipelineResult:
    """
    Run the full BJudge evaluation pipeline.

    Processes N rollouts (either live or from replay data), generates behavior
    narratives for each, then runs the comparative evaluator to select the
    best execution path.

    Args:
        objective: Natural language task instruction.
        action_sequences: For LIVE mode — list of N action sequences,
            each being a list of PyAutoGUI command strings.
        replay_data: For REPLAY mode — list of N step-data lists,
            each being a list of step dicts with image paths.
        mode: "live" for real desktop execution, "replay" for pre-recorded.
        output_dir: Base directory for visual output files.
        model: VLM model slug for narrative generation and evaluation.
        max_tokens: Maximum tokens per VLM call.

    Returns:
        A PipelineResult with all rollout results and the winner.

    Raises:
        ValueError: If the mode is invalid or required data is missing.
    """
    print(f"\n{'='*60}")
    print(f"BJudge Pipeline — {mode.upper()} mode")
    print(f"Objective: {objective}")
    print(f"{'='*60}\n")

    rollout_results: List[RolloutResult] = []

    if mode == "live":
        if not action_sequences:
            raise ValueError(
                "action_sequences is required for LIVE mode."
            )
        n_rollouts = len(action_sequences)
        print(f"Running {n_rollouts} live rollout(s)...\n")

        for i, actions in enumerate(action_sequences):
            result = run_live_rollout(
                rollout_id=i,
                action_sequence=actions,
                output_dir=output_dir,
                model=model,
                max_tokens=max_tokens,
            )
            rollout_results.append(result)
            print(f"  ✓ Rollout {i} complete: {len(actions)} steps\n")

    elif mode == "replay":
        if not replay_data:
            raise ValueError(
                "replay_data is required for REPLAY mode."
            )
        n_rollouts = len(replay_data)
        print(f"Processing {n_rollouts} replay rollout(s)...\n")

        for i, steps_data in enumerate(replay_data):
            result = run_replay_rollout(
                rollout_id=i,
                steps_data=steps_data,
                model=model,
                max_tokens=max_tokens,
            )
            rollout_results.append(result)
            print(f"  ✓ Rollout {i} complete: {len(steps_data)} steps\n")

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'live' or 'replay'.")

    # ---------------------------------------------------------------------------
    # Comparative Evaluation
    # ---------------------------------------------------------------------------
    narratives = [r.narrative for r in rollout_results]

    if len(narratives) == 1:
        # Only one rollout — skip evaluation, it wins by default
        print("Single rollout — skipping comparative evaluation.\n")
        return PipelineResult(
            objective=objective,
            rollouts=rollout_results,
            winner_index=0,
            evaluator_thoughts="Single rollout — no comparison needed.",
        )

    print(f"Running Comparative Behavior Evaluation ({len(narratives)} candidates)...\n")
    eval_result = evaluate(
        objective=objective,
        narratives=narratives,
        model=model,
        max_tokens=max_tokens,
    )

    winner_idx = eval_result["winner_index"]
    rollout_results[winner_idx].score = 1.0

    pipeline_result = PipelineResult(
        objective=objective,
        rollouts=rollout_results,
        winner_index=winner_idx,
        evaluator_thoughts=eval_result["thoughts"],
    )

    print(f"{'='*60}")
    print(f"WINNER: Rollout #{winner_idx} (1-indexed: {eval_result['winner_1indexed']})")
    print(f"{'='*60}\n")

    return pipeline_result


# ---------------------------------------------------------------------------
# Result Serialization
# ---------------------------------------------------------------------------

def save_results(result: PipelineResult, output_path: str) -> str:
    """
    Save pipeline results to a JSON file.

    Args:
        result: The PipelineResult to serialize.
        output_path: Destination file path.

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = {
        "objective": result.objective,
        "winner_index": result.winner_index,
        "evaluator_thoughts": result.evaluator_thoughts,
        "rollouts": [],
    }

    for r in result.rollouts:
        rollout_data = {
            "rollout_id": r.rollout_id,
            "narrative": r.narrative,
            "score": r.score,
            "steps": r.trajectory.to_dict(),
            "step_narratives": [
                {
                    "step_number": sn.step_number,
                    "thoughts": sn.thoughts,
                    "facts": sn.facts,
                }
                for sn in r.step_narratives
            ],
        }
        data["rollouts"].append(rollout_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BJudge Orchestrator")
    print("=" * 40)
    print()
    print("Usage:")
    print("  from src.orchestrator import run_pipeline, save_results")
    print()
    print("  # Live mode:")
    print("  result = run_pipeline(")
    print('      objective="Open Settings and enable Dark Mode",')
    print("      action_sequences=[")
    print('          ["pyautogui.click(50, 750)", "pyautogui.click(200, 300)"],')
    print('          ["pyautogui.click(100, 750)", "pyautogui.click(300, 200)"],')
    print("      ],")
    print('      mode="live",')
    print("  )")
    print()
    print("  # Replay mode:")
    print("  result = run_pipeline(")
    print('      objective="Open Settings and enable Dark Mode",')
    print("      replay_data=[rollout_0_steps, rollout_1_steps],")
    print('      mode="replay",')
    print("  )")
    print()
    print('  save_results(result, "data/results.json")')
