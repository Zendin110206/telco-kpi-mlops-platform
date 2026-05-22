from __future__ import annotations

from fastapi import APIRouter

from telco_kpi_mlops.services.metadata_service import (
    build_model_metadata,
    model_metadata_as_dict,
)

router = APIRouter(prefix="/model", tags=["metadata"])


@router.get("/info")
def model_info() -> dict[str, object]:
    metadata = build_model_metadata()
    return model_metadata_as_dict(metadata)
