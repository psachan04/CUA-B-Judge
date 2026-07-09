"""
BJudge Pipeline — Narrative Generator (Module B)

Implements the transition fact extraction function:

    ϕᵢ = G(sᵢ, aᵢ, sᵢ₊₁)

For each step in a trajectory, this module sends the annotated "before"
screenshot, the action command string, and the zoomed "after" crop to the
orchestrator VLM (z-ai/glm-5.2) via OpenRouter. The VLM returns structured
<thoughts>/<answer> blocks describing the precise environmental state changes
induced by the action.

The module then aggregates per-step facts into a single chronological
Behavior Narrative string for the full trajectory.
"""

from typing import List, Optional

from src.config import ORCHESTRATOR_MODEL, ORCHESTRATOR_MAX_TOKENS
from src.api_client import call_vlm, parse_thoughts, parse_answer_facts
from src.trajectory_models import (
    Step,
    StepNarrative,
    Trajectory,
    RolloutResult,
)


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

STEP_NARRATIVE_SYSTEM_PROMPT: str = """You are a senior workspace visualization utility and a precise UI state-change observer.

You will be shown:
1. A text description of a programmatic GUI action (PyAutoGUI syntax).
2. An annotated screenshot taken BEFORE the action was executed, with the target coordinates marked by a colored circle.
3. A zoomed 200×200 crop of the area around the action target taken AFTER the action was executed.

Your task is to identify and report the EXACT environmental state changes that occurred as a direct result of this action.

You MUST structure your response using these EXACT block delimiters:

<thoughts>
Provide your step-level reasoning here. Describe what you observe in the before screenshot, identify the UI element at the marked coordinates, and analyze what changed in the after crop. Be thorough but concise.
</thoughts>

<answer>
- [First precise environmental change]
- [Second precise environmental change]
- [Additional changes as needed]
</answer>

CRITICAL RULES:
- The <answer> block must contain ONLY a markdown unordered list of objective, factual environmental state changes.
- Do NOT include system clock changes, cursor position changes, or other incidental OS state noise.
- Do NOT include speculative or assumed changes — only report what is directly observable.
- Each bullet point must be a complete, standalone factual statement."""


# ---------------------------------------------------------------------------
# Per-Step Narrative Generation
# ---------------------------------------------------------------------------

def generate_step_narrative(
    before_screenshot_path: str,
    action_command: str,
    after_crop_path: str,
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
    step_number: int = 0,
) -> StepNarrative:
    """
    Generate a narrative for a single action step using the VLM.

    Sends a multimodal prompt with the before screenshot, the action
    description, and the after crop to extract transition facts.

    Args:
        before_screenshot_path: Path to the annotated "before" screenshot.
        action_command: The PyAutoGUI action string (e.g., "pyautogui.click(500, 300)").
        after_crop_path: Path to the 200×200 zoomed crop taken after the action.
        model: VLM model slug for the API call.
        max_tokens: Maximum response tokens.
        step_number: The ordinal step number in the trajectory.

    Returns:
        A StepNarrative dataclass with parsed thoughts, facts, and raw response.
    """
    text_content = (
        f"Action executed: {action_command}\n\n"
        f"The first image is the annotated BEFORE screenshot showing the "
        f"target coordinates. The second image is the zoomed AFTER crop "
        f"showing the result of the action."
    )

    raw_response = call_vlm(
        model=model,
        system_prompt=STEP_NARRATIVE_SYSTEM_PROMPT,
        text_content=text_content,
        image_paths=[before_screenshot_path, after_crop_path],
        max_tokens=max_tokens,
    )

    thoughts = parse_thoughts(raw_response)
    facts = parse_answer_facts(raw_response)

    return StepNarrative(
        step_number=step_number,
        thoughts=thoughts,
        facts=facts,
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Trajectory-Level Narrative Aggregation
# ---------------------------------------------------------------------------

def generate_trajectory_narrative(
    trajectory: Trajectory,
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
) -> RolloutResult:
    """
    Generate a complete Behavior Narrative for an entire trajectory.

    Iterates through each step, calls the VLM for fact extraction, and
    aggregates the results into a single chronological narrative string.

    Prerequisites:
        Each Step in the trajectory must have its annotated_image_path and
        crop_image_path fields populated (by the visual engine).

    Args:
        trajectory: A Trajectory object with fully processed steps.
        model: VLM model slug.
        max_tokens: Maximum tokens per VLM call.

    Returns:
        A RolloutResult containing the trajectory, per-step narratives,
        and the aggregated behavior narrative string.
    """
    step_narratives: List[StepNarrative] = []
    narrative_lines: List[str] = []

    for step in trajectory.steps:
        # Validate that visual processing has been done
        if not step.annotated_image_path or not step.crop_image_path:
            raise ValueError(
                f"Step {step.step_number} is missing visual engine outputs. "
                f"Run the visual engine before narrative generation."
            )

        sn = generate_step_narrative(
            before_screenshot_path=step.annotated_image_path,
            action_command=step.action_description,
            after_crop_path=step.crop_image_path,
            model=model,
            max_tokens=max_tokens,
            step_number=step.step_number,
        )

        step_narratives.append(sn)

        # Store narrative text on the Step object as well
        facts_text = "; ".join(sn.facts) if sn.facts else "(no observable changes)"
        step.narrative = facts_text

        # Build the aggregated narrative line
        narrative_lines.append(
            f"Step {step.step_number} [{step.action_description}]: {facts_text}"
        )

    aggregated_narrative = "\n".join(narrative_lines)

    return RolloutResult(
        rollout_id=trajectory.rollout_id,
        trajectory=trajectory,
        narrative=aggregated_narrative,
        step_narratives=step_narratives,
    )


# ---------------------------------------------------------------------------
# Offline / Replay Variant
# ---------------------------------------------------------------------------

def generate_narrative_from_paths(
    rollout_id: int,
    steps_data: List[dict],
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
) -> RolloutResult:
    """
    Generate a trajectory narrative from pre-recorded screenshot paths.

    This is the replay mode variant — no live OS interaction needed. Each
    entry in steps_data must be a dict with keys:
        - step_number (int)
        - action_description (str)
        - before_image_path (str)
        - after_image_path (str)
        - annotated_image_path (str)
        - crop_image_path (str)

    Args:
        rollout_id: Integer ID for this rollout.
        steps_data: List of step dictionaries with image paths.
        model: VLM model slug.
        max_tokens: Maximum tokens per VLM call.

    Returns:
        A RolloutResult with the complete behavior narrative.
    """
    steps = [
        Step(
            step_number=sd["step_number"],
            before_image_path=sd["before_image_path"],
            action_description=sd["action_description"],
            after_image_path=sd["after_image_path"],
            annotated_image_path=sd["annotated_image_path"],
            crop_image_path=sd["crop_image_path"],
        )
        for sd in steps_data
    ]

    trajectory = Trajectory(rollout_id=rollout_id, steps=steps)
    return generate_trajectory_narrative(
        trajectory=trajectory,
        model=model,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Narrative Generator — Module B")
    print("=" * 40)
    print()
    print("This module requires live OpenRouter API access to run.")
    print("To test, ensure OPENROUTER_API_KEY is set and provide")
    print("real screenshot paths.")
    print()
    print("Example usage:")
    print("  from src.narrative_generator import generate_step_narrative")
    print("  sn = generate_step_narrative(")
    print('      before_screenshot_path="data/visual_outputs/step1_annotated.png",')
    print('      action_command="pyautogui.click(500, 300)",')
    print('      after_crop_path="data/visual_outputs/step1_crop.png",')
    print("  )")
    print("  print(sn.facts)")