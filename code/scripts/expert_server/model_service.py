#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import time
import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import gc

from lang_sam import LangSAM


class ImageRequest(BaseModel):
    image: str  # base64-encoded RGB image
    text: str   # prompt text


class PredictionResponse(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    masks: List[List[List[int]]]


class BatchProcessor:
    """
    Simple batched inference queue. Requests are accumulated and processed
    in batches up to `max_batch_size`. Results are delivered via Futures.
    """
    def __init__(self, max_batch_size: int = 8, max_queue_size: int = 100) -> None:
        self.model = LangSAM(
            sam_type="sam2.1_hiera_small",
            ckpt_path_sam="replace/with/your/model_path",
            ckpt_path_gdino="replace/with/your/model_path",
        )
        self.max_batch_size = max_batch_size
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
        self.start_time = time.time()
        self.port = args.port  # set after argparse

    async def _print_queue_stats(self) -> None:
        """Periodic service heartbeat and queue metrics."""
        while True:
            uptime = time.time() - self.start_time
            qsize = self.request_queue.qsize()
            app.state.logger.info("Port=%s | Uptime=%.2fs | Queue=%d", self.port, uptime, qsize)
            await asyncio.sleep(60)

    async def add_request(self, image: Image.Image, text: str) -> Dict[str, Any]:
        """Enqueue a request and await its result."""
        if self.request_queue.full():
            raise HTTPException(status_code=503, detail="Server queue is full")

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.request_queue.put((image, text, fut))

        if not self.processing:
            asyncio.create_task(self._process_batch())

        try:
            return await asyncio.wait_for(fut, timeout=3000.0)
        except asyncio.TimeoutError as e:
            raise HTTPException(status_code=504, detail="Processing timeout") from e

    async def _process_batch(self) -> None:
        """Drain the queue in batches and run model inference."""
        self.processing = True
        try:
            while not self.request_queue.empty():
                batch_images: List[Image.Image] = []
                batch_texts: List[str] = []
                batch_futures: List[asyncio.Future] = []

                # Build a batch (non-blocking); task_done/join are not used.
                while len(batch_images) < self.max_batch_size and not self.request_queue.empty():
                    try:
                        image, text, fut = self.request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if not fut.cancelled():
                        batch_images.append(image)
                        batch_texts.append(text)
                        batch_futures.append(fut)

                if not batch_images:
                    continue

                try:
                    with torch.no_grad():
                        results = self.model.predict(
                            images_pil=batch_images,
                            texts_prompt=batch_texts,
                            box_threshold=0.3,
                            text_threshold=0.25,
                        )

                    for fut, res in zip(batch_futures, results):
                        fut.set_result(
                            {
                                "boxes": res["boxes"].tolist() if len(res["boxes"]) else [],
                                "labels": res["labels"] if len(res["labels"]) else [],
                                "masks": res["masks"].tolist() if len(res["masks"]) else [],
                            }
                        )

                    del results
                    gc.collect()
                    app.state.logger.info(
                        "Batch processed | size=%d | remaining=%d",
                        len(batch_images),
                        self.request_queue.qsize(),
                    )
                except Exception as e:
                    error_dir = "error_logs"
                    os.makedirs(error_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                    err_path = os.path.join(error_dir, f"error_{ts}.txt")
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(f"Error: {str(e)}\n\nTexts:\n")
                        for i, text in enumerate(batch_texts):
                            f.write(f"{i}: {text}\n")

                    for i, img in enumerate(batch_images):
                        img.save(os.path.join(error_dir, f"error_{ts}_img_{i}.jpg"))

                    app.state.logger.error("Batch failed; logs saved to %s", error_dir)
                    for fut in batch_futures:
                        if not fut.done():
                            fut.set_exception(e)
        finally:
            self.processing = False


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000, help="Service port")
parser.add_argument("--max_batch_size", type=int, default=8, help="Max batch size")
args = parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = logging.getLogger("uvicorn")
    stats_task = asyncio.create_task(processor._print_queue_stats())
    try:
        yield
    finally:
        stats_task.cancel()


app = FastAPI(lifespan=lifespan)
processor = BatchProcessor(max_batch_size=args.max_batch_size, max_queue_size=10000)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    try:
        img_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        result = await processor.add_request(image, request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(
        "model_service:app",
        host="0.0.0.0",
        port=args.port,
        limit_concurrency=10000,
        backlog=10000,
        log_level="debug",
    )
