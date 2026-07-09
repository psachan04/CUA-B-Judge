"""
BJudge Pipeline — Comparative Behavior Evaluator (Phase 3)

Implements the MCQ-style tournament selector. Given N trajectory behavior
narratives and the original task objective, this module constructs a
multi-choice comparison prompt and calls the orchestrator VLM (z-ai/glm-5.2)
to vote on the single best execution path.

The evaluator enforces the <thoughts>/<answer> output schema:
  - <thoughts>: Deep multi-trajectory comparative analysis
  - <answer>: A single integer (1-indexed) identifying the winning rollout
"""

import re
from typing import List, Optional

from src.config import ORCHESTRATOR_MODEL, ORCHESTRATOR_MAX_TOKENS
from src.api_client import call_text_llm, parse_thoughts, parse_answer_int


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT: str = """You are a Comparative Behavior Evaluator for computer-use agent trajectories.

You will receive:
1. A task OBJECTIVE that the agent was instructed to accomplish.
2. N numbered CANDIDATE BEHAVIOR NARRATIVES — each is a chronological text summary of what a separate execution attempt (rollout) actually did on the desktop.

Your task is to determine which rollout BEST accomplished the stated objective.

You MUST structure your response using these EXACT block delimiters:

<thoughts>
Provide a deep, multi-trajectory comparative analysis here. For each candidate:
- Assess whether its actions logically progress toward the objective
- Identify any errors, dead ends, or signs of trajectory drift
- Note any evidence of successful task completion
- Compare relative quality across all candidates
Be thorough and systematic in your evaluation matrix.
</thoughts>

<answer>
[INTEGER]
</answer>

CRITICAL RULES:
- The <answer> block must contain EXACTLY ONE integer — the 1-indexed number of the best rollout.
- If multiple rollouts appear equally good, choose the one that completed with fewer unnecessary actions.
- If NO rollout succeeded, choose the one that made the most meaningful progress.
- Do NOT output anything after the </answer> tag."""


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_comparative_prompt(
    objective: str,
    narratives: List[str],
) -> str:
    """
    Build the MCQ-style evaluation prompt with N candidate narratives.

    Args:
        objective: The natural language task instruction.
        narratives: List of N behavior narrative strings (one per rollout).

    Returns:
        The formatted user message for the evaluation VLM call.

    Raises:
        ValueError: If fewer than 1 narrative is provided.
    """
    if not narratives:
        raise ValueError("At least one narrative is required for evaluation.")

    candidate_blocks = []
    for i, narrative in enumerate(narratives, start=1):
        candidate_blocks.append(
            f"--- Candidate {i} ---\n{narrative}\n--- End Candidate {i} ---"
        )

    candidates_text = "\n\n".join(candidate_blocks)

    prompt = (
        f"OBJECTIVE:\n{objective}\n\n"
        f"CANDIDATE BEHAVIOR NARRATIVES ({len(narratives)} total):\n\n"
        f"{candidates_text}\n\n"
        f"Which candidate rollout best accomplishes the objective? "
        f"Analyze all candidates and select the best one."
    )

    return prompt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    objective: str,
    narratives: List[str],
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
) -> dict:
    """
    Run the comparative evaluation to select the best rollout.

    Sends the MCQ prompt to the VLM and parses the structured response
    to extract the winner index and reasoning trace.

    Args:
        objective: The task instruction string.
        narratives: List of N behavior narrative strings.
        model: VLM model slug for evaluation.
        max_tokens: Maximum response tokens.

    Returns:
        Dictionary with keys:
            - "winner_index": int (0-indexed)
            - "winner_1indexed": int (1-indexed, as returned by the LLM)
            - "thoughts": str (evaluator's reasoning)
            - "raw_response": str (full VLM response)

    Raises:
        ValueError: If the VLM response cannot be parsed.
        RuntimeError: If the API call fails.
    """
    user_message = build_comparative_prompt(objective, narratives)

    raw_response = call_text_llm(
        model=model,
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=max_tokens,
    )

    thoughts = parse_thoughts(raw_response)
    winner_1indexed = parse_answer_int(raw_response)

    # Validate the index is within range
    if winner_1indexed < 1 or winner_1indexed > len(narratives):
        raise ValueError(
            f"Evaluator returned winner index {winner_1indexed}, but only "
            f"{len(narratives)} candidates were provided."
        )

    return {
        "winner_index": winner_1indexed - 1,  # 0-indexed for internal use
        "winner_1indexed": winner_1indexed,
        "thoughts": thoughts,
        "raw_response": raw_response,
    }


# ---------------------------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------------------------

def select_best_rollout(
    objective: str,
    narratives: List[str],
    model: str = ORCHESTRATOR_MODEL,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
) -> int:
    """
    Simplified interface that returns only the 0-indexed winner.

    Args:
        objective: Task instruction.
        narratives: List of N behavior narratives.
        model: VLM model slug.
        max_tokens: Maximum response tokens.

    Returns:
        0-indexed integer of the best rollout.
    """
    result = evaluate(objective, narratives, model, max_tokens)
    return result["winner_index"]


# ---------------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Comparative Behavior Evaluator — Phase 3")
    print("=" * 50)

    # Demo: build a prompt with mock narratives (no API call)
    mock_objective = "Open the Settings app and enable Dark Mode."
    mock_narratives = [
        (
            "Step 1 [pyautogui.click(50, 750)]: Dock icon clicked, Settings app launched.\n"
            "Step 2 [pyautogui.click(200, 300)]: Display settings panel opened.\n"
            "Step 3 [pyautogui.click(400, 250)]: Dark Mode toggle switched ON. "
            "Screen theme changed to dark."
        ),
        (
            "Step 1 [pyautogui.click(50, 750)]: Dock icon clicked, Safari launched instead.\n"
            "Step 2 [pyautogui.click(300, 50)]: URL bar focused.\n"
            "Step 3 [pyautogui.hotkey('command', 'w')]: Safari window closed. "
            "Desktop is now visible."
        ),
    ]

    prompt = build_comparative_prompt(mock_objective, mock_narratives)
    print("\n--- Generated MCQ Prompt ---")
    print(prompt)
    print("--- End Prompt ---\n")

    # Simulate a parsed response
    mock_response = (
        "<thoughts>\n"
        "Candidate 1 correctly opened Settings, navigated to Display, and "
        "toggled Dark Mode. Candidate 2 launched the wrong application "
        "(Safari) and never reached Settings.\n"
        "</thoughts>\n"
        "<answer>1</answer>"
    )
    thoughts = parse_thoughts(mock_response)
    winner = parse_answer_int(mock_response)
    print(f"Thoughts: {thoughts}")
    print(f"Winner (1-indexed): {winner}")
    print(f"Winner (0-indexed): {winner - 1}")