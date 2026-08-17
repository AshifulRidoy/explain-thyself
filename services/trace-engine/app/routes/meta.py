"""Metadata routes: /health and /models."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    from ..models.registry import MODEL_REGISTRY, backend_status

    spec = MODEL_REGISTRY.get(settings.model)
    status = backend_status()
    return {
        "status": "ok",
        "model": settings.model,
        "modelRegistered": spec is not None,
        "device": status.device,
        "loaded": status.loaded,
        "selfCheck": status.self_check,
    }


@router.get("/models")
async def models() -> dict:
    from ..models.registry import MODEL_REGISTRY, backend_status

    status = backend_status()
    return {
        "active": settings.model,
        "device": status.device,
        "loaded": status.loaded,
        "models": [
            {
                "key": spec.key,
                "hfId": spec.hf_id,
                "layerCount": spec.layer_count,
                "headCount": spec.head_count,
                "dModel": spec.d_model,
                "paramCount": spec.param_count,
                "dtype": spec.dtype,
            }
            for spec in MODEL_REGISTRY.values()
        ],
    }
