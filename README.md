# BJudge — Test-Time Scaling for Computer-Use Agents

An implementation of the test-time scaling pipeline described in *Scaling Agents for Computer Use* (arXiv:2510.02250).

## Problem

Traditional Computer-Use Agents (CUAs) are brittle on long-horizon tasks — a single misclick or UI mismatch compounds into catastrophic trajectory drift. BJudge fixes this by shifting from single-trajectory evaluation to **wide-scaling test-time compute**: run N parallel rollouts, compress each into a text-based Behavior Narrative, and let a VLM judge vote on the best path.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BJudge Pipeline                       │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Visual   │──▶│  Narrative   │──▶│  Comparative   │  │
│  │  Engine   │   │  Generator   │   │  Evaluator     │  │
│  │ Module A  │   │  Module B    │   │  Phase 3       │  │
│  └──────────┘   └──────────────┘   └────────────────┘  │
│       │                │                    │           │
│  Screenshot      VLM Fact            MCQ Tournament     │
│  Capture +       Extraction          Selection →        │
│  Retina Scale    ϕᵢ = G(sᵢ,aᵢ,sᵢ₊₁)  Winner Index     │
│  + Action Mark                                         │
│  + 200×200 Crop                                        │
└─────────────────────────────────────────────────────────┘
```

The pipeline operates in three stages:

1. **Visual Engine** (`visual_engine.py`) — Captures desktop screenshots, applies Retina ×2 coordinate scaling, overlays action-type markers (click/move/drag), and extracts boundary-clamped 200×200 focus crops.

2. **Narrative Generator** (`narrative_generator.py`) — Sends before/after screenshot pairs to a Vision-Language Model to extract objective environmental state changes as structured facts.

3. **Comparative Evaluator** (`evaluator.py`) — Stacks N trajectory narratives as MCQ candidates and selects the best execution path via structured `<thoughts>`/`<answer>` reasoning.

## Models (OpenRouter)

| Role | Model | Purpose |
|------|-------|---------|
| Orchestrator / Judge | `z-ai/glm-5.2` | Narrative extraction, trajectory evaluation |
| Worker / Coder | `deepseek/deepseek-v4-flash` | Code generation tasks |

## Project Structure

```
src/
├── config.py              # Centralized constants & environment
├── api_client.py          # OpenRouter multimodal API client
├── visual_engine.py       # Module A — capture → scale → mark → crop
├── action_marker.py       # Pillow circle/label drawer
├── zoom_crop.py           # 200×200 boundary-clamped crop
├── narrative_generator.py # Module B — VLM fact extraction
├── trajectory_models.py   # Dataclasses (Step, Trajectory, RolloutResult, etc.)
├── evaluator.py           # Phase 3 — MCQ comparative evaluator
├── orchestrator.py        # Top-level pipeline controller
├── main.py                # CLI entry point
└── agent_workspace.py     # Git worktree sandbox for code generation
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set your API key
export OPENROUTER_API_KEY='your-key-here'

# 3. Run in replay mode (pre-recorded screenshots)
uv run python src/main.py \
    --objective "Open Settings and enable Dark Mode" \
    --replay data/replay.json \
    --mode replay

# 4. Run in live mode (real desktop actions)
uv run python src/main.py \
    --objective "Open Settings and enable Dark Mode" \
    --actions data/actions.json \
    --mode live
```

## Input File Formats

**Actions JSON** (live mode):
```json
[
  ["pyautogui.click(50, 750)", "pyautogui.click(200, 300)"],
  ["pyautogui.click(100, 750)", "pyautogui.click(300, 200)"]
]
```

**Replay JSON** (replay mode):
```json
[
  [
    {
      "step_number": 1,
      "action_description": "pyautogui.click(500, 300)",
      "before_image_path": "data/rollout_0/step_001/before.png",
      "after_image_path": "data/rollout_0/step_001/after.png",
      "annotated_image_path": "data/rollout_0/step_001/annotated.png",
      "crop_image_path": "data/rollout_0/step_001/crop.png"
    }
  ]
]
```

## Environment Requirements

- **OS:** macOS (Darwin)
- **Display:** High-DPI Retina (logical coordinates scaled ×2)
- **Python:** ≥ 3.14
- **API:** OpenRouter API key with access to `z-ai/glm-5.2`
