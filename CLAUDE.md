# CLAUDE.md

## Build, Run & Test Commands
* **Install Dependencies:** `uv sync`
* **Run Full Pipeline (replay):** `python -m bjudge --objective "..." --replay data/replay.json --mode replay`
* **Run Full Pipeline (live):** `python -m bjudge --objective "..." --actions data/actions.json --mode live`
* **Run Visual Engine Self-Test:** `uv run python -m bjudge.vision.engine`
* **Run Evaluator Self-Test:** `uv run python -m bjudge.evaluation.comparator`
* **Run Data Models Self-Test:** `uv run python -m bjudge.models`
* **Compile-Check All Modules:** `find bjudge -name '*.py' -exec uv run python -m py_compile {} \;`

---

## Coding Guidelines & Invariants

### 1. Package Structure
* **Main package:** `bjudge/` — all pipeline code lives here.
* **Sub-packages:** `core/`, `vision/`, `narrative/`, `evaluation/`, `pipeline/` — one per architectural layer.
* **Dev utilities:** `tools/` — not part of the pipeline, not imported by `bjudge/`.
* **Imports:** Always use fully qualified paths (e.g., `from bjudge.vision.engine import ...`). Never use relative imports.

### 2. Environment & Hardware Constraints
* **Operating System:** macOS (Darwin).
* **Retina Scale Factor:** High-DPI screens double pixel density. All logical layout coordinates tracked or handled by actions must be multiplied by a scale factor of `2` before being drawn on or cropped from physical image buffers.
* **UI Latency Sleep:** Always enforce a strict 3-second delay after executing a GUI action before capturing the subsequent ("after") screenshot to account for rendering latency.

### 3. Pillow (PIL) Processing Rules
* **Boundary Clamping:** When slicing a $200\times200$ square box around action targets, explicitly wrap image bounds using `max(0, ...)` and `min(dimension, ...)` to ensure edge-clicks do not cause geometry crashes.
* **Action Markings:** Maintain uniform visual marker color schemes:
  * Clicks: Solid red circle labeled `Click`
  * Movements: Solid blue circle labeled `MoveTo`
  * Drags: Blue circle (`MoveTo`) to green circle (`DragTo`) connected by a green line.

### 4. LLM/Inference Layer Constraints
* **OpenRouter Billing Override:** When interfacing with the Orchestrator (`z-ai/glm-5.2`), always include an explicit `max_tokens=4096` override configuration header to clear OpenRouter pre-flight credit checks without generating billing limit blocks.
* **Type Hinting:** Enforce strict PEP 484 type hints across all functional entry points (`Tuple`, `Optional`, `List`, `Dict`).

### 5. Output Parsing Protocol
All generative logic handled by agent components must output and consume strict XML/regex-parseable block delimiters:
* Step Fact Generation: Must split into `<thoughts>` (reasoning matrix) and `<answer>` (unordered markdown list of precise environmental changes).
* Path Selection Evaluation: Must split into `<thoughts>` (cross-trajectory analysis) and `<answer>` (a standalone integer corresponding to the chosen execution path).

### 6. Configuration
* All constants (model slugs, crop sizes, delay timings, marker colors, API endpoints) live in `bjudge/config.py`. Do not hardcode values in other modules.
* The API client (`bjudge/core/api_client.py`) is the single gateway for all OpenRouter calls. Do not create ad-hoc API clients elsewhere.