"""
BJudge Pipeline — Unified OpenRouter API Client

Handles all LLM communication through the OpenRouter gateway. Supports both
text-only and multimodal (text + base64 image) payloads. Includes retry logic
with exponential backoff and proper max_tokens header injection for GLM-5.2
credit reservation.
"""

import base64
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from openai import OpenAI

from src.config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    THOUGHTS_OPEN,
    THOUGHTS_CLOSE,
    ANSWER_OPEN,
    ANSWER_CLOSE,
)


def _get_client() -> OpenAI:
    """Lazy-initialize the OpenRouter client."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Export it before running the pipeline."
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


# ---------------------------------------------------------------------------
# Image Encoding
# ---------------------------------------------------------------------------

def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/png")


# ---------------------------------------------------------------------------
# Core API Callers
# ---------------------------------------------------------------------------

def call_vlm(
    model: str,
    system_prompt: str,
    text_content: str,
    image_paths: Optional[List[str]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
    max_retries: int = 3,
) -> str:
    """
    Call a Vision-Language Model via OpenRouter with multimodal content.

    Args:
        model: Model slug (e.g., "z-ai/glm-5.2").
        system_prompt: System instruction string.
        text_content: The text portion of the user message.
        image_paths: Optional list of image file paths to include.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        max_retries: Number of retry attempts on transient failures.

    Returns:
        The model's response text content.
    """
    client = _get_client()

    # Build the multimodal user content array
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": text_content}
    ]

    if image_paths:
        for img_path in image_paths:
            b64 = encode_image_base64(img_path)
            media_type = _image_media_type(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64}"
                }
            })

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Retry loop with exponential backoff
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            if message.content:
                return message.content.strip()
            # Some models return reasoning in a separate field
            reasoning = getattr(message, "reasoning", None) or getattr(
                message, "reasoning_content", None
            )
            if reasoning:
                return reasoning.strip()
            return "[Empty response from model]"

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)

    raise RuntimeError(
        f"OpenRouter API call failed after {max_retries} attempts: {last_error}"
    )


def call_text_llm(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = ORCHESTRATOR_MAX_TOKENS,
    max_retries: int = 3,
) -> str:
    """
    Call a text-only LLM via OpenRouter (no images).

    Args:
        model: Model slug.
        system_prompt: System instruction string.
        user_message: The user's text message.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        max_retries: Number of retry attempts.

    Returns:
        The model's response text content.
    """
    client = _get_client()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            if message.content:
                return message.content.strip()
            reasoning = getattr(message, "reasoning", None) or getattr(
                message, "reasoning_content", None
            )
            if reasoning:
                return reasoning.strip()
            return "[Empty response from model]"

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)

    raise RuntimeError(
        f"OpenRouter API call failed after {max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Response Parsers
# ---------------------------------------------------------------------------

def parse_thoughts(response_text: str) -> str:
    """
    Extract content between <thoughts> and </thoughts> tags.

    Returns:
        The extracted thoughts string, or empty string if not found.
    """
    pattern = re.compile(
        rf"{re.escape(THOUGHTS_OPEN)}\s*(.*?)\s*{re.escape(THOUGHTS_CLOSE)}",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(response_text)
    return match.group(1).strip() if match else ""


def parse_answer(response_text: str) -> str:
    """
    Extract content between <answer> and </answer> tags.

    Returns:
        The extracted answer string, or empty string if not found.
    """
    pattern = re.compile(
        rf"{re.escape(ANSWER_OPEN)}\s*(.*?)\s*{re.escape(ANSWER_CLOSE)}",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(response_text)
    return match.group(1).strip() if match else ""


def parse_answer_int(response_text: str) -> int:
    """
    Extract an integer from inside <answer> tags.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If no valid integer is found inside <answer> tags.
    """
    answer_text = parse_answer(response_text)
    if not answer_text:
        raise ValueError(
            "Could not find content inside <answer> tags in the response."
        )
    # Extract the first integer from the answer block
    int_match = re.search(r"(\d+)", answer_text)
    if not int_match:
        raise ValueError(
            f"No integer found inside <answer> tags. Got: '{answer_text}'"
        )
    return int(int_match.group(1))


def parse_answer_facts(response_text: str) -> List[str]:
    """
    Extract a markdown unordered list from inside <answer> tags.

    Returns:
        List of fact strings (one per bullet point).
    """
    answer_text = parse_answer(response_text)
    if not answer_text:
        return []
    # Match lines starting with -, *, or • (markdown list markers)
    facts = re.findall(r"^[\-\*•]\s*(.+)$", answer_text, re.MULTILINE)
    return [f.strip() for f in facts if f.strip()]
