"""
BJudge Pipeline — Trajectory Data Models

Dataclasses representing the core data structures flowing through the
pipeline: individual steps, step-level VLM narratives, full trajectories,
and rollout results.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class Step:
    """A single action step in a trajectory."""
    step_number: int
    before_image_path: str
    action_description: str
    after_image_path: str
    # Paths populated by the visual engine
    annotated_image_path: Optional[str] = None
    crop_image_path: Optional[str] = None
    # Narrative populated by the narrative generator (VLM)
    narrative: Optional[str] = None


@dataclass
class StepNarrative:
    """
    Parsed VLM output for a single step's transition fact extraction.

    ϕᵢ = G(sᵢ, aᵢ, sᵢ₊₁)
    """
    step_number: int
    thoughts: str                    # Content from <thoughts> block
    facts: List[str]                 # Parsed bullet points from <answer> block
    raw_response: str                # Full unparsed VLM response


@dataclass
class Trajectory:
    """An ordered sequence of steps constituting one execution path."""
    rollout_id: int = 0
    steps: List[Step] = field(default_factory=list)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize all steps to a list of dictionaries."""
        return [asdict(step) for step in self.steps]


@dataclass
class RolloutResult:
    """
    The output of a single rollout after narrative generation.

    Contains the trajectory, its aggregated behavior narrative, and the
    optional evaluation score assigned by the comparative evaluator.
    """
    rollout_id: int
    trajectory: Trajectory
    narrative: str                        # Aggregated chronological behavior narrative
    step_narratives: List[StepNarrative] = field(default_factory=list)
    score: Optional[float] = None         # Set by evaluator if this rollout wins


@dataclass
class PipelineResult:
    """
    Final output of the BJudge pipeline.

    Contains all rollout results and identifies the winner selected by the
    comparative behavior evaluator.
    """
    objective: str
    rollouts: List[RolloutResult]
    winner_index: int                     # 0-indexed into rollouts list
    evaluator_thoughts: str = ""          # Reasoning trace from the evaluator

    @property
    def best_rollout(self) -> RolloutResult:
        return self.rollouts[self.winner_index]


if __name__ == "__main__":
    # ---- Demo: Create a full pipeline result with mock data ----

    step1 = Step(
        step_number=1,
        before_image_path="img_1_before.png",
        action_description="pyautogui.click(500, 300)",
        after_image_path="img_1_after.png",
        annotated_image_path="img_1_annotated.png",
        crop_image_path="img_1_crop.png",
        narrative="Clicked the 'Submit' button. The form was submitted.",
    )

    sn1 = StepNarrative(
        step_number=1,
        thoughts="The submit button is centered in the dialog.",
        facts=["Form submission dialog closed", "Success banner appeared"],
        raw_response="<thoughts>...</thoughts><answer>- Form submission dialog closed\n- Success banner appeared</answer>",
    )

    traj = Trajectory(rollout_id=0, steps=[step1])
    rollout = RolloutResult(
        rollout_id=0,
        trajectory=traj,
        narrative="Step 1: Clicked Submit. Form closed, success banner appeared.",
        step_narratives=[sn1],
    )

    result = PipelineResult(
        objective="Submit the contact form",
        rollouts=[rollout],
        winner_index=0,
        evaluator_thoughts="Only one rollout, it succeeded.",
    )

    print(f"Winner: Rollout #{result.winner_index}")
    print(f"Narrative: {result.best_rollout.narrative}")
    print(f"Steps: {[asdict(s) for s in result.best_rollout.trajectory.steps]}")