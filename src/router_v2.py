# --- src/agent_workspace.py (updated) ---
import pathlib
import subprocess
import re
from pathlib import Path

class WorkspaceManager:
    def __init__(self, repo_root: str = ".", worktree_base_dir: str = "../worktrees"):
        self.repo_root = Path(repo_root).resolve()
        self.worktree_base_dir = Path(worktree_base_dir).resolve()
        self.current_worktree_path = None
        self.current_branch = None

    def create_task_branch(self, task_name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', task_name.replace(' ', '-'))
        branch_name = f"feature/{sanitized}"
        worktree_path = self.worktree_base_dir / sanitized
        self.worktree_base_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=str(self.repo_root), capture_output=True, text=True)
        subprocess.run(["git", "worktree", "add", str(worktree_path), branch_name], cwd=str(self.repo_root), capture_output=True, text=True)
        self.current_worktree_path = worktree_path
        self.current_branch = branch_name
        return str(worktree_path)

    def write_and_validate_code(self, filename: str, code: str) -> dict | str:
        if self.current_worktree_path is None:
            raise RuntimeError("No active worktree. Call create_task_branch first.")
        file_path = self.current_worktree_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code)
        result = subprocess.run(["python", "-c", f"import ast; ast.parse(open('{file_path}').read())"], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "code_snapshot": code, "file_path": str(file_path)}
        else:
            return result.stderr.strip()

    def teardown(self, task_name: str) -> None:
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', task_name.replace(' ', '-'))
        branch_name = f"feature/{sanitized}"
        worktree_path = self.worktree_base_dir / sanitized
        if worktree_path.exists():
            subprocess.run(
                ["git", "worktree", "remove", "-f", str(worktree_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True
            )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True
        )
        if self.current_worktree_path == worktree_path:
            self.current_worktree_path = None
            self.current_branch = None

if __name__ == "__main__":
    manager = WorkspaceManager()
    try:
        path = manager.create_task_branch("dummy-test-task")
        print(f"Worktree created at: {path}")
        result = manager.write_and_validate_code("test_script.py", "x = 1")
        print(f"Validation result: {result}")
    finally:
        manager.teardown("dummy-test-task")
        print("Cleanup done.")

# --- src/router_v2.py ---
import os
import re
import sys
import openai
from src.agent_workspace import WorkspaceManager

# Check API key at startup
if os.environ.get("OPENROUTER_API_KEY") is None:
    print("Error: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

ORCHESTRATOR_SYS_PROMPT = """You are the Principal Systems Architect for a Python multi-agent framework. 
            Engage with the human developer. If their request is ambiguous, ASK clarifying questions. 
            If the request is clear, draft a strict, edge-case-proof technical specification 
            that a subordinate coding agent can follow to write the code."""
WORKER_SYS_PROMPT = """You are an automated code generation assistant. You receive specifications and output 
        production-ready Python code. CRITICAL: Output ONLY valid Python code enclosed in a single 
        markdown code block (```python ... ```). No explanations."""

ORCHESTRATOR_MODEL = "z-ai/glm-5.2"
WORKER_MODEL = "deepseek/deepseek-v4-flash"


def query_llm(messages: list, model_slug: str = ORCHESTRATOR_MODEL) -> str:
    """Handles the direct API call to OpenRouter."""
    try:
        response = client.chat.completions.create(
            model=model_slug,
            messages=messages,
            temperature=0.2,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[API Connection Error]: {str(e)}"


def execute_worker_task(task_name: str, specification: str) -> dict | None:
    """Manages the isolated workspace and the 3-strike self-healing validation loop."""
    manager = WorkspaceManager()
    try:
        manager.create_task_branch(task_name)
        worker_history = [
            {"role": "system", "content": WORKER_SYS_PROMPT},
            {"role": "user", "content": f"Write Python code for this exact spec:\n\n{specification}"}
        ]

        # The Self-Healing Loop (Max 3 Attempts)
        for attempt in range(3):
            print(f"  -> Worker Attempt {attempt + 1}/3...")
            response_text = query_llm(worker_history, model_slug=WORKER_MODEL)
            worker_history.append({"role": "assistant", "content": response_text})

            # Extract the python code using regex
            match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
            if not match:
                worker_history.append({
                    "role": "user",
                    "content": "You did not output a valid python code block. Please try again."
                })
                continue

            code = match.group(1).strip()

            # Validate the code in the sandbox
            result = manager.write_and_validate_code("generated_script.py", code)

            if isinstance(result, dict) and result.get("status") == "success":
                return result  # Validation passed!
            else:
                # Validation failed, feed the traceback back to the worker
                worker_history.append({
                    "role": "user",
                    "content": f"The code failed to compile with the following error:\n{result}\nPlease fix the syntax error and output the complete corrected code."
                })

        return None  # Failed after 3 attempts
    except Exception as e:
        print(f"Task execution failed: {e}")
        return None
    finally:
        print("Cleaning up workspace...")
        manager.teardown(task_name)

if __name__ == "__main__":
    print("===================================================")
    print(" BJudge Agentic Workspace v2 (Type 'exit' to quit)")
    print(" Type 'END' on a new line to submit your prompt.")
    print("===================================================\n")

    orchestrator_history = [{"role": "system", "content": ORCHESTRATOR_SYS_PROMPT}]

    while True:
        print("\n[You] > ")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
            except EOFError:
                break

        user_input = "\n".join(lines).strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down BJudge Workspace. Goodbye!")
            break
        if not user_input:
            # If the user just hits ENTER, trigger the worker execution based on the last spec
            if len(orchestrator_history) < 2:
                print("No specification provided yet. Talk to the Orchestrator first.")
                continue

            last_spec = orchestrator_history[-1]["content"]
            task_name = "auto-task"  # You can make this dynamic later

            print(f"\n[Worker Coder is spinning up workspace and writing code...]")
            result = execute_worker_task(task_name, last_spec)

            if result:
                print("\n Validation succeeded! Code compiled in isolated workspace.")
                final_path = input("Enter final file path to save (e.g., src/trajectory_models.py) or ENTER to skip > ")
                if final_path.strip():
                    try:
                        pathlib.Path(final_path).write_text(result["code_snapshot"], encoding="utf-8")
                        print(f" Success!")
                    except Exception as e:
                        print(f" Error saving final file: {e}")
            else:
                print("\n Worker failed to generate valid code after 3 attempts.")
            continue

        # Normal Orchestrator Conversation
        orchestrator_history.append({"role": "user", "content": user_input})
        print(f"\n[Orchestrator is thinking...]")

        # Make sure this uses whatever query function your script generated
        response_text = query_llm(orchestrator_history)
        print(f"\n=== Orchestrator Response ===\n{response_text}\n=============================\n")
        orchestrator_history.append({"role": "assistant", "content": response_text})
        print(
            "[System] Press ENTER on a blank line (then 'END') to send this to the Worker, or type feedback to revise.")