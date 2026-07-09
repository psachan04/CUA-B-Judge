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
│  │  Vision   │──▶│  Narrative   │──▶│  Evaluation    │  │
│  │  Engine   │   │  Generator   │   │  Comparator    │  │
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

## Project Structure

```
bjudge/                        # Main Python package
├── __init__.py                # Package root
├── __main__.py                # CLI entry point (python -m bjudge)
├── config.py                  # Centralized constants & environment
├── models.py                  # Dataclasses (Step, Trajectory, RolloutResult, etc.)
│
├── core/                      # Infrastructure
│   └── api_client.py          # OpenRouter multimodal API client + response parsers
│
├── vision/                    # Module A — visual pre-processing
│   ├── engine.py              # Full capture → scale → mark → crop pipeline
│   ├── markers.py             # Pillow circle/label action marker drawing
│   └── crop.py                # 200×200 boundary-clamped zoom crop
│
├── narrative/                 # Module B — VLM fact extraction
│   └── generator.py           # Step-level & trajectory-level narrative generation
│
├── evaluation/                # Phase 3 — trajectory comparison
│   └── comparator.py          # MCQ tournament evaluator
│
└── pipeline/                  # Orchestration
    └── orchestrator.py        # Top-level N-rollout controller + result serialization

tools/                         # Dev utilities (not part of pipeline)
└── workspace.py               # Git worktree sandbox for code generation

data/
└── visual_outputs/            # Generated screenshots and crops

paper/                         # Reference paper (arXiv:2510.02250)
```

## Models (OpenRouter)

| Role | Model | Purpose |
|------|-------|---------|
| Orchestrator / Judge | `z-ai/glm-5.2` | Narrative extraction, trajectory evaluation |
| Worker / Coder | `deepseek/deepseek-v4-flash` | Code generation tasks |

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set your API key
export OPENROUTER_API_KEY='your-key-here'

# 3. Run in replay mode (pre-recorded screenshots)
python -m bjudge \
    --objective "Open Settings and enable Dark Mode" \
    --replay data/replay.json \
    --mode replay

# 4. Run in live mode (real desktop actions)
python -m bjudge \
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
