import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, validator

from middleware.auth_middleware import get_current_user
from services.bg_service import BGServiceError, clear_bg_model_cache, process_bg_removal
from services.rate_limiter import limiter
from services.task_claim_check import store_image_claim_check

try:
    from worker import bg_remove_task
except Exception:
    bg_remove_task = None

print("BG_REMOVER LOADED")

router = APIRouter()


class BGRemoveRequest(BaseModel):
    image_base64: str

    @validator("image_base64")
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid image data")
        return v


def clear_model_cache() -> None:
    clear_bg_model_cache()


@router.post("/remove-bg")
@limiter.limit("6/minute")
async def remove_background(request: Request, payload: BGRemoveRequest, user=Depends(get_current_user)):
    try:
        return await asyncio.to_thread(process_bg_removal, payload.image_base64)
    except BGServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.post("/remove-bg/async", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("6/minute")
async def remove_background_async(request: Request, payload: BGRemoveRequest, user=Depends(get_current_user)):
    if bg_remove_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        image_url = store_image_claim_check(
            payload.image_base64,
            user_id=user.get("user_id", "anonymous"),
            task_type="bg_remove",
        )
        task = bg_remove_task.delay(image_url)
        return {
            "success": True,
            "status": "queued",
            "task_id": task.id,
            "task_type": "bg_remove_task",
            "image_claim_check_url": image_url,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue bg removal: {exc}")
