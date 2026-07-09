"""
BJudge Pipeline — Centralized Configuration

All environment constants, model slugs, API settings, and visual processing
parameters referenced by the pipeline modules. Import from here instead of
hardcoding values.
"""

import os
from typing import Tuple

# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------------------
# Model Slugs
# ---------------------------------------------------------------------------
# Vision model — must support image inputs (multimodal).
# Used by the narrative generator to analyze before/after screenshots.
VISION_MODEL: str = "google/gemini-2.5-flash"

# Orchestrator / evaluator model — text-only, used for comparative evaluation.
ORCHESTRATOR_MODEL: str = "z-ai/glm-5.2"

# Worker model for code generation tasks.
WORKER_MODEL: str = "deepseek/deepseek-v4-flash"

# GLM-5.2 requires an explicit max_tokens override to pass OpenRouter
# pre-flight credit reservation validation checks.
ORCHESTRATOR_MAX_TOKENS: int = 4096
DEFAULT_TEMPERATURE: float = 0.2

# ---------------------------------------------------------------------------
# Display / Retina
# ---------------------------------------------------------------------------
RETINA_SCALE_FACTOR: int = 2  # macOS High-DPI logical→physical multiplier

# ---------------------------------------------------------------------------
# Visual Engine
# ---------------------------------------------------------------------------
CROP_SIZE: int = 200  # pixels — boundary-clamped zoom crop box

# Action marker colours (RGBA)
MARKER_COLOR_CLICK: Tuple[int, int, int, int] = (255, 0, 0, 255)       # Red
MARKER_COLOR_MOVETO: Tuple[int, int, int, int] = (0, 100, 255, 255)    # Blue
MARKER_COLOR_DRAGTO: Tuple[int, int, int, int] = (0, 200, 0, 255)      # Green
MARKER_OUTLINE_COLOR: Tuple[int, int, int, int] = (255, 255, 255, 255) # White
MARKER_RADIUS: int = 8  # pixels — outer radius of action marker circle

# ---------------------------------------------------------------------------
# Pipeline Timing
# ---------------------------------------------------------------------------
POST_ACTION_DELAY: float = 3.0  # seconds to wait after action before capture

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
VISUAL_OUTPUTS_DIR: str = os.path.join(DATA_DIR, "visual_outputs")

# ---------------------------------------------------------------------------
# LLM Output Parsing — Block Delimiter Tags
# ---------------------------------------------------------------------------
THOUGHTS_OPEN: str = "<thoughts>"
THOUGHTS_CLOSE: str = "</thoughts>"
ANSWER_OPEN: str = "<answer>"
ANSWER_CLOSE: str = "</answer>"
