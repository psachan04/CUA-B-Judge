"""
BJudge Pipeline — Visual Engine (Module A)

Deterministic visual pre-processor implementing the full pipeline:

    Capture Screenshot → Save Raw PNG → Retina Scale ×2
        → Draw Action Marker → Extract 200×200 Crop → Return File Paths

Composes the action_marker (red dot drawer) and zoom_crop modules into a
single-call pipeline stage. Supports both live PyAutoGUI capture and
pre-recorded screenshot replay for testing.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional, Literal

from PIL import Image, ImageDraw, ImageFont

from bjudge.config import (
    RETINA_SCALE_FACTOR,
    CROP_SIZE,
    MARKER_RADIUS,
    MARKER_COLOR_CLICK,
    MARKER_COLOR_MOVETO,
    MARKER_COLOR_DRAGTO,
    MARKER_OUTLINE_COLOR,
    POST_ACTION_DELAY,
    VISUAL_OUTPUTS_DIR,
)
from bjudge.vision.markers import draw_action_marker
from bjudge.vision.crop import zoom_crop


# Type alias for supported action types
ActionType = Literal["click", "moveto", "dragto"]


# ---------------------------------------------------------------------------
# Screenshot Capture
# ---------------------------------------------------------------------------

def capture_screenshot(output_path: str) -> str:
    """
    Capture a full desktop screenshot and save as PNG.

    Uses PyAutoGUI as the primary capture method. Falls back to the native
    macOS ``screencapture -x`` command if PyAutoGUI fails (e.g., due to
    missing screen-recording permissions).

    Args:
        output_path: Destination file path for the raw PNG.

    Returns:
        The absolute path to the saved screenshot.

    Raises:
        RuntimeError: If both capture methods fail.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Primary: PyAutoGUI
    try:
        import pyautogui
        screenshot = pyautogui.screenshot()
        screenshot.save(output_path)
        return os.path.abspath(output_path)
    except Exception:
        pass

    # Fallback: macOS native screencapture (silent mode)
    try:
        result = subprocess.run(
            ["screencapture", "-x", output_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and os.path.isfile(output_path):
            return os.path.abspath(output_path)
    except Exception:
        pass

    raise RuntimeError(
        f"Failed to capture screenshot. Both PyAutoGUI and screencapture -x "
        f"failed. Ensure screen-recording permissions are granted."
    )


# ---------------------------------------------------------------------------
# Retina Coordinate Scaling
# ---------------------------------------------------------------------------

def apply_retina_scaling(
    logical_x: int,
    logical_y: int,
    scale_factor: int = RETINA_SCALE_FACTOR,
) -> Tuple[int, int]:
    """
    Convert logical (PyAutoGUI) coordinates to physical pixel coordinates.

    On macOS High-DPI Retina displays, the physical pixel buffer is 2× the
    logical resolution. All image-level operations (drawing, cropping) must
    use scaled coordinates.

    Args:
        logical_x: Logical x coordinate from PyAutoGUI.
        logical_y: Logical y coordinate from PyAutoGUI.
        scale_factor: Pixel density multiplier (default: 2 for Retina).

    Returns:
        Tuple of (physical_x, physical_y).
    """
    return logical_x * scale_factor, logical_y * scale_factor


# ---------------------------------------------------------------------------
# Action Marker Drawing (Extended)
# ---------------------------------------------------------------------------

def _get_marker_color(action_type: ActionType) -> Tuple[int, int, int, int]:
    """Return the RGBA fill color for a given action type."""
    color_map = {
        "click": MARKER_COLOR_CLICK,
        "moveto": MARKER_COLOR_MOVETO,
        "dragto": MARKER_COLOR_DRAGTO,
    }
    return color_map.get(action_type, MARKER_COLOR_CLICK)


def _get_marker_label(action_type: ActionType) -> str:
    """Return the text label for a given action type."""
    label_map = {
        "click": "Click",
        "moveto": "MoveTo",
        "dragto": "DragTo",
    }
    return label_map.get(action_type, "Click")


def draw_extended_marker(
    image_path: str,
    action_type: ActionType,
    physical_x: int,
    physical_y: int,
    output_path: str,
    drag_target: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Draw an action-type-aware marker on a screenshot.

    For 'click' and 'moveto': draws a solid circle at the target coordinates
    with the appropriate color and a text label.

    For 'dragto': draws a blue circle at the initial position (drag_target
    should be the starting point), a green circle at the target position,
    and a connecting green vector line.

    Args:
        image_path: Path to the source screenshot.
        action_type: One of "click", "moveto", "dragto".
        physical_x: Physical x coordinate (Retina-scaled).
        physical_y: Physical y coordinate (Retina-scaled).
        output_path: Destination path for annotated image.
        drag_target: For drag actions, the (x, y) destination of the drag.

    Returns:
        Path to the saved annotated image.
    """
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    fill_color = _get_marker_color(action_type)
    label = _get_marker_label(action_type)
    r = MARKER_RADIUS

    if action_type == "dragto" and drag_target is not None:
        # Draw origin marker (blue — MoveTo)
        _draw_circle(draw, physical_x, physical_y, r,
                     MARKER_COLOR_MOVETO, MARKER_OUTLINE_COLOR)
        _draw_label(draw, physical_x, physical_y, r, "MoveTo")

        # Draw destination marker (green — DragTo)
        dx, dy = drag_target
        _draw_circle(draw, dx, dy, r,
                     MARKER_COLOR_DRAGTO, MARKER_OUTLINE_COLOR)
        _draw_label(draw, dx, dy, r, "DragTo")

        # Draw connecting vector line (green)
        draw.line(
            [(physical_x, physical_y), (dx, dy)],
            fill=MARKER_COLOR_DRAGTO,
            width=2,
        )
    else:
        # Single marker for click or moveto
        _draw_circle(draw, physical_x, physical_y, r,
                     fill_color, MARKER_OUTLINE_COLOR)
        _draw_label(draw, physical_x, physical_y, r, label)

    # Save (handle JPEG alpha channel limitation)
    output_lower = output_path.lower()
    if output_lower.endswith((".jpg", ".jpeg")):
        img = img.convert("RGB")
    img.save(output_path)
    return os.path.abspath(output_path)


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int],
) -> None:
    """Draw a filled circle with a contrasting outline."""
    # Outer (outline)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=outline,
    )
    # Inner (fill, 2px smaller)
    inner_r = radius - 2
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        fill=fill,
    )


def _draw_label(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    label: str,
) -> None:
    """Draw a text label below the action marker."""
    text_y = cy + radius + 4
    try:
        draw.text((cx - 10, text_y), label, fill=(255, 255, 255, 255))
    except Exception:
        pass  # Skip label if font rendering fails


# ---------------------------------------------------------------------------
# Main Pipeline Entry Point
# ---------------------------------------------------------------------------

def annotate_and_crop(
    screenshot_path: str,
    action_type: ActionType,
    logical_x: int,
    logical_y: int,
    output_dir: str,
    step_id: str = "step",
    drag_target_logical: Optional[Tuple[int, int]] = None,
) -> Tuple[str, str]:
    """
    Full visual pre-processing pipeline for a single action step.

    1. Apply Retina coordinate scaling (×2).
    2. Draw action marker on the screenshot.
    3. Extract a boundary-clamped 200×200 crop centered on the action target.

    Args:
        screenshot_path: Path to the raw screenshot PNG.
        action_type: One of "click", "moveto", "dragto".
        logical_x: Logical x coordinate from PyAutoGUI.
        logical_y: Logical y coordinate from PyAutoGUI.
        output_dir: Directory to save annotated and cropped images.
        step_id: Identifier string used in output filenames.
        drag_target_logical: For drag actions, (logical_x, logical_y) of drag
            destination.

    Returns:
        Tuple of (annotated_image_path, cropped_image_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Scale logical → physical coordinates
    phys_x, phys_y = apply_retina_scaling(logical_x, logical_y)

    drag_target_phys = None
    if drag_target_logical is not None:
        drag_target_phys = apply_retina_scaling(*drag_target_logical)

    # 2. Draw action marker
    annotated_path = os.path.join(output_dir, f"{step_id}_annotated.png")
    draw_extended_marker(
        image_path=screenshot_path,
        action_type=action_type,
        physical_x=phys_x,
        physical_y=phys_y,
        output_path=annotated_path,
        drag_target=drag_target_phys,
    )

    # 3. Extract boundary-clamped crop centered on the action target
    crop_img = zoom_crop(
        image=annotated_path,
        x=phys_x,
        y=phys_y,
        crop_size=CROP_SIZE,
    )
    crop_path = os.path.join(output_dir, f"{step_id}_crop.png")
    crop_img.save(crop_path)

    return os.path.abspath(annotated_path), os.path.abspath(crop_path)


def capture_and_process(
    action_type: ActionType,
    logical_x: int,
    logical_y: int,
    output_dir: str,
    step_id: str = "step",
    drag_target_logical: Optional[Tuple[int, int]] = None,
) -> Tuple[str, str, str]:
    """
    Live capture + full visual pre-processing in a single call.

    Captures a live screenshot, then runs the annotate_and_crop pipeline.

    Args:
        action_type: One of "click", "moveto", "dragto".
        logical_x: Logical x coordinate.
        logical_y: Logical y coordinate.
        output_dir: Directory for all output files.
        step_id: Step identifier for filenames.
        drag_target_logical: For drag actions, destination coordinates.

    Returns:
        Tuple of (raw_screenshot_path, annotated_path, crop_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, f"{step_id}_raw.png")
    capture_screenshot(raw_path)

    annotated_path, crop_path = annotate_and_crop(
        screenshot_path=raw_path,
        action_type=action_type,
        logical_x=logical_x,
        logical_y=logical_y,
        output_dir=output_dir,
        step_id=step_id,
        drag_target_logical=drag_target_logical,
    )

    return raw_path, annotated_path, crop_path


# ---------------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("Visual Engine — Self-Test")
    print("=" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a synthetic 800x600 desktop image
        sample_path = os.path.join(tmpdir, "desktop.png")
        sample = Image.new("RGB", (800, 600), (40, 40, 50))
        # Add some fake UI elements
        draw = ImageDraw.Draw(sample)
        draw.rectangle([50, 50, 250, 80], fill=(60, 60, 80))
        draw.rectangle([300, 200, 500, 240], fill=(70, 100, 70))
        sample.save(sample_path)

        # Test: Click at logical (150, 65) → physical (300, 130)
        ann_path, crop_path = annotate_and_crop(
            screenshot_path=sample_path,
            action_type="click",
            logical_x=150,
            logical_y=65,
            output_dir=tmpdir,
            step_id="test_click",
        )
        print(f"✓ Click annotated: {ann_path}")
        print(f"✓ Click crop:      {crop_path}")

        # Verify crop dimensions
        crop = Image.open(crop_path)
        print(f"  Crop size: {crop.size[0]}×{crop.size[1]}")

        # Test: Drag from (100, 100) to (250, 120)
        ann_drag, crop_drag = annotate_and_crop(
            screenshot_path=sample_path,
            action_type="dragto",
            logical_x=100,
            logical_y=100,
            output_dir=tmpdir,
            step_id="test_drag",
            drag_target_logical=(250, 120),
        )
        print(f"✓ Drag annotated:  {ann_drag}")
        print(f"✓ Drag crop:       {crop_drag}")

    print("\nAll visual engine tests passed.")
