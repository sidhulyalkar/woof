"""Private FastAPI worker for Woof Behavior Vision.

Run separately from the main compatibility service because video dependencies and latency profiles are
very different. Raw uploads are held only in request memory by this worker. Production deployments
should put the worker on a private network and configure WOOF_BEHAVIOR_SERVICE_TOKEN.
"""

from __future__ import annotations

import hmac
import json
import os
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from .adapters import MediaInput
from .contracts import ContractError, RequestMetadata
from .pipeline import BehaviorVisionPipeline

MAX_MEDIA_BYTES = 50 * 1024 * 1024
ALLOWED_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "video/mp4",
    "video/webm",
    "video/quicktime",
}

app = FastAPI(
    title="Woof Behavior Vision",
    version="1.0.0-beta.1",
    docs_url=None if os.getenv("WOOF_BEHAVIOR_DISABLE_DOCS") == "1" else "/docs",
)
pipeline = BehaviorVisionPipeline()


def _authorize(authorization: str | None) -> None:
    expected = os.getenv("WOOF_BEHAVIOR_SERVICE_TOKEN", "").strip()
    if not expected:
        return
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="missing behavior service bearer token")
    supplied = authorization[len(prefix) :]
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="invalid behavior service bearer token")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "service": "woof-behavior-vision",
        "enabledAdapters": [getattr(adapter, "adapter_id", "unknown") for adapter in pipeline.adapters],
        "authoritativeEmotionInference": False,
        "automaticGreetingRecommendation": False,
    }


@app.post("/v1/behavior/analyze")
async def analyze(
    metadata: Annotated[str, Form()],
    media: Annotated[UploadFile, File()],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    _authorize(authorization)

    if media.content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="unsupported behavior media type")

    try:
        raw_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
    if not isinstance(raw_metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")

    try:
        request_metadata = RequestMetadata.from_json(raw_metadata)
    except ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = await media.read(MAX_MEDIA_BYTES + 1)
    if len(payload) > MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail="behavior media exceeds 50 MB")
    if not payload:
        raise HTTPException(status_code=400, detail="behavior media is empty")

    analysis = pipeline.analyze(
        MediaInput(
            bytes=payload,
            mime_type=media.content_type or "application/octet-stream",
            filename=media.filename or "behavior-media",
        ),
        request_metadata,
    )
    return analysis.to_api()
