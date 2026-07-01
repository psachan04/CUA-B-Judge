import os
import re
from openai import OpenAI

# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

ORCHESTRATOR = "z-ai/glm-5.2"
WORKER_CODER = "deepseek/deepseek-v4-flash"


def call_agent(model_slug: str, system_prompt: str, messages: list, temperature: float = 0.2) -> str:
    """Generic function to call agents with conversation history."""
    try:
        api_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=model_slug,
            messages=api_messages,
            temperature=temperature,
            max_tokens=4096
        )
        message = response.choices[0].message
        if message.content:
            return message.content.strip()
        reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None)
        if reasoning:
            return f"[Internal Reasoning]:\n{reasoning.strip()}"
        return "[Error: Empty response]"
    except Exception as e:
        return f"API Connection Error: {str(e)}"


def extract_python_code(text: str) -> str:
    """Extracts raw python code from markdown code blocks."""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def main_loop():
    print("===================================================")
    print(" BJudge Agentic CLI Initialized (Type 'exit' to quit)")
    print(" Orchestrator: GLM-5.2 | Worker: DeepSeek-V4-Flash")
    print("===================================================\n")

    orchestrator_system = (
        "You are the Principal Systems Architect for a Python multi-agent framework. "
        "Engage with the human developer. If their request is ambiguous, ASK clarifying questions. "
        "If the request is clear, draft a strict, edge-case-proof technical specification "
        "that a subordinate coding agent can follow to write the code."
    )

    worker_system = (
        "You are an automated code generation assistant. You receive specifications and output "
        "production-ready Python code. CRITICAL: Output ONLY valid Python code enclosed in a single "
        "markdown code block (```python ... ```). No explanations."
    )
 
    chat_history = []

    while True:
        print(
            "\n[You] > (Paste your prompt. When finished, type 'END' on a new blank line and press Enter. Type 'exit' to quit.)")
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
            print("Shutting down BJudge Agentic Loop. Goodbye!")
            break
        if not user_input:
            continue

        # 1. Talk to Orchestrator
        chat_history.append({"role": "user", "content": user_input})
        print(f"\n[Orchestrator ({ORCHESTRATOR}) is thinking...]")

        orchestrator_reply = call_agent(ORCHESTRATOR, orchestrator_system, chat_history)
        print(f"\n=== Orchestrator Response ===\n{orchestrator_reply}\n=============================\n")
        chat_history.append({"role": "assistant", "content": orchestrator_reply})

        # 2. Human-in-the-Loop Gate
        action = input("[System] Press ENTER to send this to the Worker Coder, or type feedback to revise > ")

        if action.strip() == "":
            # 3. Trigger Worker
            print(f"\n[Worker Coder ({WORKER_CODER}) is writing code...]")
            worker_messages = [
                {"role": "user", "content": f"Write Python code for this exact spec:\n\n{orchestrator_reply}"}]
            coder_reply = call_agent(WORKER_CODER, worker_system, worker_messages)

            clean_code = extract_python_code(coder_reply)
            print("\n=== Code Generated ===")
            print(clean_code[:300] + "\n... [Code Truncated for Display] ...")

            # 4. Save File
            filename = input("\n[System] Enter file path to save (e.g., src/narrative_generator.py) or 'skip' > ")
            if filename.lower() != 'skip' and filename.strip() != "":
                try:
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(clean_code)
                    print(f" Success!")
                except Exception as e:
                    print(f" Error writing file: {e}")
        else:
            # Treat action as conversational feedback to the Orchestrator
            print("[System] Passing feedback back to Orchestrator...")
            chat_history.append({"role": "user", "content": action})

            # Instantly get revised spec
            print(f"\n[Orchestrator ({ORCHESTRATOR}) is revising...]")
            revised_reply = call_agent(ORCHESTRATOR, orchestrator_system, chat_history)
            print(f"\n=== Revised Orchestrator Response ===\n{revised_reply}\n=====================================\n")
            chat_history.append({"role": "assistant", "content": revised_reply})
            print("[System] Revision complete. Please initiate the next command to trigger the coder.")


if __name__ == "__main__":
    main_loop()