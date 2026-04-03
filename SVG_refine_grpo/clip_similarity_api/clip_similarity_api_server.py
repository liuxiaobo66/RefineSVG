#!/usr/bin/env python3
"""
CLIP 图像相似度单卡服务（支持高并发 + 微批处理）。

特性：
- 单 GPU 常驻模型（openai/clip-vit-large-patch14 视觉分支）
- 请求级高并发，服务端自动将跨请求样本合并成 micro-batch 提升吞吐
- 支持两种输入：本地路径 / base64
- 返回每对图像 cosine similarity（[-1, 1]）
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection


# =========================
# 配置（可通过环境变量覆盖）
# =========================
MODEL_PATH = os.getenv(
    "MODEL_PATH", "${CLIP_MODEL_PATH}"
)
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", "8"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "4096"))
MAX_PAIRS_PER_REQUEST = int(os.getenv("MAX_PAIRS_PER_REQUEST", "256"))
USE_FAST_PROCESSOR = os.getenv("USE_FAST_PROCESSOR", "1") == "1"


# =========================
# 输入输出协议
# =========================
class ImagePair(BaseModel):
    # 路径输入
    image_a_path: Optional[str] = None
    image_b_path: Optional[str] = None

    # base64 输入
    image_a_b64: Optional[str] = None
    image_b_b64: Optional[str] = None

    @model_validator(mode="after")
    def _validate_one_input_mode(self) -> "ImagePair":
        has_path = bool(self.image_a_path) and bool(self.image_b_path)
        has_b64 = bool(self.image_a_b64) and bool(self.image_b_b64)
        if has_path == has_b64:
            raise ValueError(
                "Each pair must provide exactly one mode: "
                "(image_a_path + image_b_path) OR (image_a_b64 + image_b_b64)."
            )
        return self


class SimilarityRequest(BaseModel):
    pairs: list[ImagePair] = Field(default_factory=list)
    normalize_to_01: bool = False


class PairResult(BaseModel):
    similarity: Optional[float] = None
    error: Optional[str] = None


class SimilarityResponse(BaseModel):
    results: list[PairResult]
    model_path: str
    device: str
    queue_size: int
    batch_size_last: int
    request_time_ms: float


@dataclass
class QueueItem:
    pair: ImagePair
    fut: asyncio.Future


# =========================
# 全局状态
# =========================
app = FastAPI(title="CLIP Similarity API", version="1.0.0")

model: CLIPVisionModelWithProjection | None = None
processor: Any = None
queue: asyncio.Queue[QueueItem] | None = None
worker_task: asyncio.Task | None = None

stats = {
    "requests_total": 0,
    "pairs_total": 0,
    "batches_total": 0,
    "pairs_failed": 0,
    "batch_size_last": 0,
}


# =========================
# 工具函数
# =========================
def _decode_b64_image(b64_text: str) -> Image.Image:
    # 兼容 data:image/png;base64,.... 格式
    if "," in b64_text and b64_text.strip().startswith("data:"):
        b64_text = b64_text.split(",", 1)[1]
    raw = base64.b64decode(b64_text)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _load_image_from_path(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        return im.convert("RGB")


def _load_pair_images(pair: ImagePair) -> tuple[Image.Image, Image.Image]:
    if pair.image_a_path and pair.image_b_path:
        return _load_image_from_path(pair.image_a_path), _load_image_from_path(pair.image_b_path)
    if pair.image_a_b64 and pair.image_b_b64:
        return _decode_b64_image(pair.image_a_b64), _decode_b64_image(pair.image_b_b64)
    raise ValueError("Invalid pair input mode.")


def _infer_pairs_from_images(image_pairs: list[tuple[Image.Image, Image.Image]]) -> list[float]:
    """对 N 对图像做一次批量推理，返回 N 个相似度。"""
    assert model is not None and processor is not None

    images: list[Image.Image] = []
    for a, b in image_pairs:
        images.extend([a, b])

    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        embeds = model(pixel_values=pixel_values).image_embeds  # [2N, D]
        embeds = F.normalize(embeds, dim=-1)
        embeds = embeds.view(len(image_pairs), 2, -1)  # [N,2,D]
        sims = (embeds[:, 0] * embeds[:, 1]).sum(dim=-1)

    return sims.float().cpu().tolist()


async def _batch_worker() -> None:
    """从队列聚合请求并微批推理。"""
    assert queue is not None

    timeout_s = BATCH_TIMEOUT_MS / 1000.0
    loop = asyncio.get_running_loop()

    while True:
        first = await queue.get()
        items = [first]
        deadline = loop.time() + timeout_s

        while len(items) < MAX_BATCH_SIZE:
            remain = deadline - loop.time()
            if remain <= 0:
                break
            try:
                nxt = await asyncio.wait_for(queue.get(), timeout=remain)
                items.append(nxt)
            except asyncio.TimeoutError:
                break

        stats["batches_total"] += 1
        stats["batch_size_last"] = len(items)

        # 并行 decode（线程池），坏样本单独报错，不影响同 batch 其他样本
        decode_results = await asyncio.gather(
            *(asyncio.to_thread(_load_pair_images, it.pair) for it in items),
            return_exceptions=True,
        )

        valid_indices: list[int] = []
        valid_images: list[tuple[Image.Image, Image.Image]] = []

        for i, (item, decoded) in enumerate(zip(items, decode_results)):
            if isinstance(decoded, Exception):
                stats["pairs_failed"] += 1
                if not item.fut.done():
                    item.fut.set_result(PairResult(similarity=None, error=str(decoded)))
            else:
                valid_indices.append(i)
                valid_images.append(decoded)

        if valid_images:
            try:
                sims = _infer_pairs_from_images(valid_images)
                for idx, sim in zip(valid_indices, sims):
                    item = items[idx]
                    if not item.fut.done():
                        item.fut.set_result(PairResult(similarity=float(sim), error=None))
            except Exception as e:
                err = f"batch inference failed: {type(e).__name__}: {e}"
                for idx in valid_indices:
                    item = items[idx]
                    stats["pairs_failed"] += 1
                    if not item.fut.done():
                        item.fut.set_result(PairResult(similarity=None, error=err))

        for _ in items:
            queue.task_done()


# =========================
# FastAPI 生命周期
# =========================
@app.on_event("startup")
async def _startup() -> None:
    global model, processor, queue, worker_task

    dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32

    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, dtype=dtype)
    model.to(DEVICE)
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=USE_FAST_PROCESSOR)

    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    worker_task = asyncio.create_task(_batch_worker())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global worker_task
    if worker_task is not None:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


# =========================
# API
# =========================
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": model is not None,
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "queue_size": 0 if queue is None else queue.qsize(),
        "stats": stats,
    }


@app.post("/v1/similarity", response_model=SimilarityResponse)
async def similarity(req: SimilarityRequest) -> SimilarityResponse:
    global queue
    if queue is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    if not req.pairs:
        raise HTTPException(status_code=400, detail="pairs must not be empty.")
    if len(req.pairs) > MAX_PAIRS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"pairs per request exceeds MAX_PAIRS_PER_REQUEST={MAX_PAIRS_PER_REQUEST}",
        )

    t0 = time.perf_counter()
    stats["requests_total"] += 1
    stats["pairs_total"] += len(req.pairs)

    loop = asyncio.get_running_loop()
    futs: list[asyncio.Future] = []

    for pair in req.pairs:
        fut = loop.create_future()
        item = QueueItem(pair=pair, fut=fut)
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="Server busy: queue is full.")
        futs.append(fut)

    raw_results = await asyncio.gather(*futs)

    results: list[PairResult] = []
    for r in raw_results:
        if isinstance(r, PairResult):
            if r.similarity is not None and req.normalize_to_01:
                r.similarity = (r.similarity + 1.0) / 2.0
            results.append(r)
        else:
            # 理论上不会走到这里，留保险
            results.append(PairResult(similarity=None, error="unknown result type"))

    dt_ms = (time.perf_counter() - t0) * 1000.0

    return SimilarityResponse(
        results=results,
        model_path=MODEL_PATH,
        device=DEVICE,
        queue_size=queue.qsize(),
        batch_size_last=stats["batch_size_last"],
        request_time_ms=round(dt_ms, 3),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "clip_similarity_api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "18080")),
        workers=1,
        log_level="info",
    )
