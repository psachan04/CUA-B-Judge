"""
UI Action Marker Image Annotation

Annotates a base image with a bright red circular marker surrounded by a
2-pixel white contrasting outline to simulate a UI interaction (e.g., a
mouse click) at a specific coordinate.
"""

from PIL import Image, ImageDraw, UnidentifiedImageError


def draw_action_marker(
    base_image_path: str,
    click_coord: tuple,
    output_path: str,
    marker_radius: int = 8,
) -> bool:
    """
    Draw a bright red circular action marker with a white contrasting outline
    on a base image at the specified click coordinate.

    The marker consists of two concentric circles:
      - Outer circle: White (255, 255, 255, 255) with radius = marker_radius
      - Inner circle: Bright Red (255, 0, 0, 255) with radius = marker_radius - 2

    This ensures the marker is visible on both dark and light backgrounds.

    Args:
        base_image_path: Path to the input image file.
        click_coord: A tuple or list of exactly two integers (x, y) representing
            the click coordinate.
        output_path: Path where the annotated image will be saved.
        marker_radius: Radius of the outer marker circle in pixels. Must be at
            least 3 to accommodate the 2-pixel contrasting outline. Defaults to 8.

    Returns:
        True if the image was successfully annotated and saved.
        False if a runtime I/O error occurred (file not found, invalid image
        format, permission error, etc.).

    Raises:
        ValueError: If any input argument is invalid (programmer error).
    """

    # 1. Input Validation (Programmer Errors — raise ValueError)

    if not isinstance(base_image_path, str) or len(base_image_path) == 0:
        raise ValueError("base_image_path must be a non-empty string")

    if not isinstance(output_path, str) or len(output_path) == 0:
        raise ValueError("output_path must be a non-empty string")

    if not isinstance(click_coord, (tuple, list)) or len(click_coord) != 2:
        raise ValueError("click_coord must be a tuple of two integers (x, y)")

    x, y = click_coord
    if (
        not isinstance(x, int)
        or isinstance(x, bool)
        or not isinstance(y, int)
        or isinstance(y, bool)
    ):
        raise ValueError("click_coord must be a tuple of two integers (x, y)")

    if not isinstance(marker_radius, int) or isinstance(marker_radius, bool):
        raise ValueError(
            "marker_radius must be at least 3 to accommodate the contrasting outline"
        )

    if marker_radius <= 2:
        raise ValueError(
            "marker_radius must be at least 3 to accommodate the contrasting outline"
        )

    # 2. Image Processing (Runtime Errors — return False)

    try:
        with Image.open(base_image_path) as img:
            # Convert to RGBA for consistent color rendering regardless of
            # the source image's original mode (L, P, CMYK, etc.)
            img = img.convert("RGBA")

            # Initialize the drawing context
            draw = ImageDraw.Draw(img)

            # Calculate bounding box for the outer circle (white outline)
            x0_outer = x - marker_radius
            y0_outer = y - marker_radius
            x1_outer = x + marker_radius
            y1_outer = y + marker_radius

            # Calculate bounding box for the inner circle (red fill)
            inner_radius = marker_radius - 2
            x0_inner = x - inner_radius
            y0_inner = y - inner_radius
            x1_inner = x + inner_radius
            y1_inner = y + inner_radius

            # Draw outer white circle first (contrasting outline)
            draw.ellipse(
                [x0_outer, y0_outer, x1_outer, y1_outer],
                fill=(255, 255, 255, 255),
            )

            # Draw inner red circle second (primary marker fill)
            draw.ellipse(
                [x0_inner, y0_inner, x1_inner, y1_inner],
                fill=(255, 0, 0, 255),
            )

            # Handle JPEG output — JPEG does not support the alpha channel
            output_lower = output_path.lower()
            if output_lower.endswith(".jpg") or output_lower.endswith(".jpeg"):
                img = img.convert("RGB")

            # Save the annotated image
            img.save(output_path)

            return True

    except (FileNotFoundError, UnidentifiedImageError, OSError, Exception):
        return False


# Demonstration block

if __name__ == "__main__":
    import tempfile
    import os

    # Create a temporary directory for the demo
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample base image (100x100, dark background)
        sample_path = os.path.join(tmpdir, "sample.png")
        sample_img = Image.new("RGBA", (100, 100), (30, 30, 30, 255))
        sample_img.save(sample_path)

        # Annotate the sample image
        output_path = os.path.join(tmpdir, "annotated.png")
        success = draw_action_marker(
            base_image_path=sample_path,
            click_coord=(50, 50),
            output_path=output_path,
            marker_radius=8,
        )

        if success:
            print(f"Success: Annotated image saved to {output_path}")
        else:
            print("Failure: Could not annotate the image.")