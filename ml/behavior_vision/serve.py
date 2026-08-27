"""Private FastAPI worker for Woof Behavior Vision.

Run separately from the main compatibility service because video dependencies and latency profiles are
very different. Raw uploads are held only in request memory by this worker. Production deployments
should put the worker on a private network and configure WOOF_BEHAVIOR_SERVICE_TOKEN plus an exact
worker-owned release identity.
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
from .release import load_release_identity, require_matching_release

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


def _worker_release_or_http_error():
    try:
        release = load_release_identity()
    except ContractError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if release is None:
        raise HTTPException(status_code=503, detail="Behavior Vision worker release is not configured")
    return release


@app.get("/health")
def health() -> dict[str, object]:
    try:
        release = load_release_identity()
        release_error = None
    except ContractError as exc:
        release = None
        release_error = str(exc)
    return {
        "ok": release is not None and release_error is None,
        "service": "woof-behavior-vision",
        "enabledAdapters": [getattr(adapter, "adapter_id", "unknown") for adapter in pipeline.adapters],
        "releaseConfigured": release is not None,
        "release": release.to_api() if release is not None else None,
        "releaseError": release_error,
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
    worker_release = _worker_release_or_http_error()

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
        qualified_release = require_matching_release(
            request_metadata.expected_release,
            worker_release,
        )
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
    return analysis.to_api(qualified_release)
