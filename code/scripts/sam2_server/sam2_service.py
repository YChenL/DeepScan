#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

# ----------------------------------------------------------------------------- #
# Working directory (ensures SAM-2 package-relative paths resolve consistently) #
# ----------------------------------------------------------------------------- #
SAM2_REPO_ROOT = Path("replace/with/your/model_path")

if SAM2_REPO_ROOT.is_dir():
    os.chdir(SAM2_REPO_ROOT)
    print(f"Working directory changed to: {os.getcwd()}")
else:
    raise FileNotFoundError(f"SAM-2 repository path not found: {SAM2_REPO_ROOT}")

# ----------------------------------------------------------------------------- #
# 1) Argument parsing                                                           #
# ----------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="SAM-2 point-prompt inference server")
parser.add_argument("--ckpt", help="Path to SAM-2 checkpoint")
parser.add_argument("--cfg",  help="Path to SAM-2 config YAML")
parser.add_argument("--port", type=int, default=8000, help="Service port")
args, _ = parser.parse_known_args()

# ----------------------------------------------------------------------------- #
# 2) I/O schema                                                                 #
# ----------------------------------------------------------------------------- #
class PointPromptReq(BaseModel):
    image_b64: str
    pos: conlist(conlist(float, min_length=2, max_length=2), min_length=1) | None = None
    neg: conlist(conlist(float, min_length=2, max_length=2), min_length=1) | None = None
    multimask: bool | None = False


class PointPromptResp(BaseModel):
    mask: dict           # {"data_b64": ..., "shape": [...], "dtype": "..."}
    ious: List[float]

# ----------------------------------------------------------------------------- #
# 3) SAM-2 engine wrapper                                                       #
# ----------------------------------------------------------------------------- #
class _Sam2Engine:
    """Load-once, thread-safe SAM-2 predictor."""
    def __init__(self, cfg: str, ckpt: str) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._logger = logging.getLogger("sam2-engine")
        self._logger.info("Loading SAM-2 model from working directory: %s", os.getcwd())

        self._model = build_sam2(cfg, ckpt).cuda().eval()
        self._predictor = SAM2ImagePredictor(self._model)
        self._logger.info("SAM-2 model is ready")

    def predict(
        self,
        img_rgb: np.ndarray,
        pos_pts: Sequence[Sequence[float]] | None,
        neg_pts: Sequence[Sequence[float]] | None,
        multimask: bool,
    ) -> tuple[np.ndarray, list[float]]:
        """Run point-prompt segmentation and return the best mask and IoUs."""
        pos_arr = np.asarray(pos_pts, dtype=np.float32) if pos_pts else np.zeros((0, 2), np.float32)
        neg_arr = np.asarray(neg_pts, dtype=np.float32) if neg_pts else np.zeros((0, 2), np.float32)

        if len(pos_arr) + len(neg_arr) > 0:
            coords = np.concatenate([pos_arr, neg_arr], axis=0)
            labels = np.concatenate(
                [np.ones(len(pos_arr)), np.zeros(len(neg_arr))],
                axis=0,
            ).astype(np.int32)
        else:
            coords, labels = None, None

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._predictor.set_image(img_rgb)
            masks, ious, _ = self._predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=bool(multimask),
            )

        best_idx = int(np.argmax(ious))
        return masks[best_idx], ious.tolist()

# ----------------------------------------------------------------------------- #
# 4) Utilities                                                                  #
# ----------------------------------------------------------------------------- #
def _b64_to_rgb_ndarray(b64_str: str) -> np.ndarray:
    """Decode base64 image into an RGB uint8 ndarray."""
    try:
        buf = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(400, f"Invalid image data: {e}")


def _ndarray_to_b64_dict(arr: np.ndarray) -> dict:
    """Encode ndarray raw bytes as base64 with shape/dtype metadata."""
    return {
        "data_b64": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }

# ----------------------------------------------------------------------------- #
# 5) FastAPI app with lifespan                                                  #
# ----------------------------------------------------------------------------- #
_engine: _Sam2Engine | None = None  # initialized in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    app.state.logger = logging.getLogger("uvicorn")
    _engine = _Sam2Engine(args.cfg, args.ckpt)

    async def _tick():
        while True:
            app.state.logger.debug("SAM-2 service alive")
            await asyncio.sleep(60)

    ticker = asyncio.create_task(_tick())
    try:
        yield
    finally:
        ticker.cancel()


app = FastAPI(
    title="SAM-2 Point-Prompt Service",
    version="0.3.0",
    lifespan=lifespan,
)

# ----------------------------------------------------------------------------- #
# 6) Endpoint                                                                   #
# ----------------------------------------------------------------------------- #
@app.post("/sam2/point_predict", response_model=PointPromptResp)
async def point_predict(req: PointPromptReq):
    if _engine is None:
        raise HTTPException(503, "Engine not initialised")

    img_rgb = _b64_to_rgb_ndarray(req.image_b64)

    loop = asyncio.get_running_loop()
    try:
        best_mask, ious = await asyncio.wait_for(
            loop.run_in_executor(
                None, _engine.predict, img_rgb, req.pos, req.neg, bool(req.multimask)
            ),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Prediction timeout (300s)")

    return PointPromptResp(mask=_ndarray_to_b64_dict(best_mask), ious=ious)

# ----------------------------------------------------------------------------- #
# 7) CLI entry                                                                  #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run(
        "sam2_service:app",  # module:app
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        limit_concurrency=1000,
        backlog=1000,
    )
