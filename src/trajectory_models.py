from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Step:
    step_number: int
    before_image_path: str
    action_description: str
    after_image_path: str
    # No mutable defaults; all required positional arguments


@dataclass
class Trajectory:
    steps: List[Step] = field(default_factory=list)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [asdict(step) for step in self.steps]


if __name__ == "__main__":
    # Create two Step objects with mock data
    step1 = Step(step_number=1,
                 before_image_path="img_1_before.png",
                 action_description="Click button A",
                 after_image_path="img_1_after.png")
    step2 = Step(step_number=2,
                 before_image_path="img_2_before.png",
                 action_description="Type 'Hello'",
                 after_image_path="img_2_after.png")

    # Instantiate a Trajectory
    trajectory = Trajectory()

    # Append steps
    trajectory.steps.append(step1)
    trajectory.steps.append(step2)

    # Convert to list of dictionaries and print
    result = trajectory.to_dict()
    for item in result:
        print(item)