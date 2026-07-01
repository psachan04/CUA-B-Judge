from typing import Union
from pathlib import Path
from PIL import Image

def zoom_crop(
    image: Union[str, Path, Image.Image],
    x: int,
    y: int,
    crop_size: int = 200
) -> Image.Image:
    """
    Extracts a square crop of size crop_size centered on (x, y).
    Boundaries are clamped to image edges to prevent out-of-bounds errors.
    """
    # If image is a file path, open it
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # Ensure coordinates are integers
    x = int(x)
    y = int(y)

    # Calculate half size
    half = crop_size // 2

    # Initial bounding box
    left = x - half
    top = y - half
    right = left + crop_size
    bottom = top + crop_size

    # Clamp to image dimensions
    left = max(0, left)
    top = max(0, top)
    right = min(image.width, right)
    bottom = min(image.height, bottom)

    # Crop and return
    return image.crop((left, top, right, bottom))