import re

class LLMJudge:
    def build_mcq_prompt(self, objective: str, action_history: list, candidate_actions: list) -> str:
        history_lines = "\n".join(f"- {action}" for action in action_history)
        candidate_lines = "\n".join(f"{i+1}. {action}" for i, action in enumerate(candidate_actions))
        
        prompt = f"""You are an expert evaluator. Your task is to select the best next action given the objective and action history.

Objective: {objective}

Action History:
{history_lines}

Candidate Actions:
{candidate_lines}

First, provide your reasoning inside <thoughts> and </thoughts> tags.
Then, output the number of your chosen action (1-indexed) inside <answer> and </answer> tags."""
        
        return prompt

    def parse_llm_response(self, response_text: str) -> int:
        match = re.search(r'<answer>\s*(\d+)\s*</answer>', response_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        raise ValueError("Could not find a valid integer inside <answer> tags.")

if __name__ == "__main__":
    judge = LLMJudge()
    
    dummy_objective = "Find the user's account ID."
    dummy_history = ["Searched database for user email.", "Found email: user@example.com"]
    dummy_candidates = ["Query accounts table for user@example.com", "Log out and restart"]
    
    prompt = judge.build_mcq_prompt(dummy_objective, dummy_history, dummy_candidates)
    print(prompt)
    print()
    
    mock_response = "<thoughts>Looks like action 2 is best.</thoughts>\n<answer>2</answer>"
    result = judge.parse_llm_response(mock_response)
    print(result)