"""
QuestClass 3D Backend
FastAPI server — receives DALL-E images, runs SAM + SAM 3D, returns asset URLs.

Start:
  source .venv/bin/activate
  uvicorn main:app --reload --port 8000
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

app = FastAPI(title="QuestClass 3D", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    image_base64: str   # raw base64 PNG (no data: prefix)


class ProcessResponse(BaseModel):
    sprite_url: str          # always present — SAM-segmented transparent PNG
    ply_url: str | None      # Gaussian splat PLY, None if SAM 3D not available
    success: bool


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-object", response_model=ProcessResponse)
async def process_object(req: ProcessRequest):
    """
    Full pipeline:
      DALL-E base64 PNG → SAM segment → SAM 3D reconstruct → Firebase upload
    Returns permanent Firebase Storage URLs.
    """
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    from pipeline import process_object_image

    try:
        result = await process_object_image(req.image_base64)
        return ProcessResponse(**result)
    except Exception as e:
        logging.error(f"process_object error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-only", response_model=ProcessResponse)
async def segment_only(req: ProcessRequest):
    """
    SAM segmentation only — no SAM 3D.
    Useful for testing / when GPU unavailable.
    """
    from pipeline import process_object_image
    import pipeline

    # Patch: temporarily disable SAM 3D
    original = pipeline._sam3d_inference
    pipeline._sam3d_inference = "disabled"   # truthy → skip load attempt

    try:
        result = await process_object_image(req.image_base64)
        result["ply_url"] = None
        return ProcessResponse(**result)
    finally:
        pipeline._sam3d_inference = original
