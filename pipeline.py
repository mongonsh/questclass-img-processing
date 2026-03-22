"""
Image Processing Pipeline (PIL-based, no SAM)
  DALL-E image (base64 PNG)
    → PIL BFS white-background removal
    → Firebase Storage upload
    → returns permanent download URL
"""

import os
import io
import uuid
import base64
import asyncio
import logging
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Background removal ───────────────────────────────────────────────────────

def remove_white_background(
    image: Image.Image,
    threshold: int = 225,
    border_tolerance: int = 38,
    max_saturation_delta: int = 26,
) -> Image.Image:
    """
    Remove white / near-white background using BFS flood-fill from image edges.
    Pixels with R,G,B all >= threshold that are reachable from the border are
    made fully transparent.  Fast pure-numpy/PIL — no ML model needed.
    """
    img = image.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    h, w = data.shape[:2]

    # Sample the image border so we can handle off-white / lightly shaded
    # backgrounds that DALL-E often returns instead of pure white.
    border_pixels = np.concatenate(
        [
            data[0, :, :3],
            data[h - 1, :, :3],
            data[1:h - 1, 0, :3],
            data[1:h - 1, w - 1, :3],
        ],
        axis=0,
    ).astype(np.int16)
    border_rgb = np.median(border_pixels, axis=0)

    # Boolean mask of pixels we've already enqueued
    visited = np.zeros((h, w), dtype=bool)

    def is_bg(y: int, x: int) -> bool:
        r, g, b = (int(data[y, x, 0]), int(data[y, x, 1]), int(data[y, x, 2]))
        if r >= threshold and g >= threshold and b >= threshold:
            return True

        # Keep flood-fill limited to edge-connected pixels that still resemble
        # the sampled border color and stay close to neutral (low saturation).
        max_delta = max(abs(r - int(border_rgb[0])), abs(g - int(border_rgb[1])), abs(b - int(border_rgb[2])))
        saturation_delta = max(r, g, b) - min(r, g, b)
        return max_delta <= border_tolerance and saturation_delta <= max_saturation_delta

    queue: deque = deque()

    # Seed BFS from all four edges
    for x in range(w):
        for y_edge in (0, h - 1):
            if not visited[y_edge, x] and is_bg(y_edge, x):
                visited[y_edge, x] = True
                queue.append((y_edge, x))
    for y in range(1, h - 1):
        for x_edge in (0, w - 1):
            if not visited[y, x_edge] and is_bg(y, x_edge):
                visited[y, x_edge] = True
                queue.append((y, x_edge))

    while queue:
        y, x = queue.popleft()
        data[y, x, 3] = 0          # make transparent
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_bg(ny, nx):
                visited[ny, nx] = True
                queue.append((ny, nx))

    # Light anti-aliasing so white halos at object edges fade out instead of
    # remaining as visible fringes after compositing.
    rgb = data[:, :, :3].astype(np.int16)
    alpha = data[:, :, 3].astype(np.uint8)
    near_bg = (
        (rgb[:, :, 0] >= threshold - 18)
        & (rgb[:, :, 1] >= threshold - 18)
        & (rgb[:, :, 2] >= threshold - 18)
        & (alpha > 0)
    )
    alpha[near_bg] = np.minimum(alpha[near_bg], 140)
    data[:, :, 3] = alpha

    return Image.fromarray(data)


# ─── Firebase upload ──────────────────────────────────────────────────────────

def _firebase_bucket():
    import firebase_admin
    from firebase_admin import credentials, storage

    if not firebase_admin._apps:
        key_path = Path(__file__).parent / "serviceAccountKey.json"
        if key_path.exists():
            cred = credentials.Certificate(str(key_path))
        else:
            cred = credentials.ApplicationDefault()
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "")
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

    return storage.bucket()


def upload_bytes_to_firebase(data: bytes, storage_path: str, content_type: str) -> str:
    bucket = _firebase_bucket()
    blob = bucket.blob(storage_path)
    blob.upload_from_string(data, content_type=content_type)
    blob.make_public()
    return blob.public_url


# ─── Public entry point ───────────────────────────────────────────────────────

async def process_object_image(image_base64: str) -> dict:
    """
    Pipeline:
      1. Decode base64 PNG
      2. PIL BFS white-background removal
      3. Upload transparent PNG to Firebase Storage
    Returns: { "sprite_url": str, "success": bool }
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _process_sync, image_base64)


def _process_sync(image_base64: str) -> dict:
    uid = str(uuid.uuid4())[:8]

    # 1. Decode image
    raw = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(raw))
    logger.info(f"Processing image {image.size} …")

    # 2. PIL background removal
    sprite = remove_white_background(image)

    # 3. Encode to PNG bytes
    buf = io.BytesIO()
    sprite.save(buf, format="PNG")

    # 4. Upload to Firebase
    sprite_url = upload_bytes_to_firebase(
        buf.getvalue(),
        f"objects/{uid}-sprite.png",
        "image/png",
    )
    logger.info(f"Sprite uploaded → {sprite_url}")

    return {
        "sprite_url": sprite_url,
        "ply_url": None,
        "success": True,
    }
