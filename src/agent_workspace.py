import subprocess
import sys
import re
import shutil
from pathlib import Path


class WorkspaceManager:
    def __init__(self, repo_root: str = ".", worktree_base_dir: str = ".agent_worktrees"):
        self.repo_root = Path(repo_root).resolve()
        self.worktree_base_dir = (self.repo_root / worktree_base_dir).resolve()
        self.current_worktree_path: Path | None = None
        self.current_branch: str | None = None
        self.worktree_base_dir.mkdir(parents=True, exist_ok=True)

    def create_task_branch(self, task_name: str) -> str:
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', task_name.replace(' ', '-'))
        branch_name = f"feature/{sanitized_name}"
        worktree_path = self.worktree_base_dir / sanitized_name

        if worktree_path.exists():
            subprocess.run(
                ["git", "worktree", "remove", "-f", str(worktree_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True
            )
            if worktree_path.exists():
                shutil.rmtree(str(worktree_path), ignore_errors=True)

        cmd = ["git", "worktree", "add", str(worktree_path), "-b", branch_name]
        result = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        self.current_worktree_path = worktree_path
        self.current_branch = branch_name
        return str(self.current_worktree_path)

    def write_and_validate_code(self, file_path: str, code_contents: str) -> dict | str:
        if self.current_worktree_path is None:
            raise RuntimeError("No active worktree. Call create_task_branch first.")

        target_file = self.current_worktree_path / file_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(code_contents, encoding='utf-8')

        cmd = [sys.executable, "-m", "py_compile", str(target_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return result.stderr

        return {
            "status": "success",
            "file_path": str(target_file),
            "branch": self.current_branch,
            "code_snapshot": code_contents
        }

    def teardown(self, task_name: str) -> None:
        """Removes the git worktree and deletes the temporary task branch."""
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', task_name.replace(' ', '-'))
        branch_name = f"feature/{sanitized_name}"
        worktree_path = self.worktree_base_dir / sanitized_name

        if worktree_path.exists():
            subprocess.run(
                ["git", "worktree", "remove", "-f", str(worktree_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True
            )
            # Ensure the directory is completely wiped if git leaves anything behind
            if worktree_path.exists():
                shutil.rmtree(str(worktree_path), ignore_errors=True)

        # Force delete the temporary feature branch
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
        worktree_path = manager.create_task_branch("dummy-test-task")
        print(f"Worktree created at: {worktree_path}")

        success_result = manager.write_and_validate_code("dummy_script.py", "print('Hello World')\n")
        print("Success Test Result:", success_result)

        failure_result = manager.write_and_validate_code("bad_script.py", "print('Hello World'\n")
        print("Failure Test Result:", failure_result)
    finally:
        manager.teardown("dummy-test-task")
        print("Teardown complete.")