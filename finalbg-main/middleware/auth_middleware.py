import asyncio
import hashlib
import json
import os
import time
from threading import Lock

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.appwrite_service import build_request_account
from services.settings import get_settings


_bearer = HTTPBearer(auto_error=False)
_memory_token_cache: dict[str, tuple[float, dict]] = {}
_memory_cache_lock = Lock()
_redis_cache_lock = Lock()
_redis_client = None


def _cache_ttl_seconds() -> int:
    settings = get_settings()
    return max(5, int(settings.AUTH_TOKEN_CACHE_TTL_SECONDS))


def _token_cache_key(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"auth:token:{digest}"


def _get_cached_user_from_memory(cache_key: str) -> dict | None:
    now = time.time()
    with _memory_cache_lock:
        entry = _memory_token_cache.get(cache_key)
        if not entry:
            return None
        expires_at, payload = entry
        if expires_at <= now:
            _memory_token_cache.pop(cache_key, None)
            return None
        return payload


def _set_cached_user_in_memory(cache_key: str, payload: dict, ttl_seconds: int) -> None:
    with _memory_cache_lock:
        _memory_token_cache[cache_key] = (time.time() + ttl_seconds, payload)


def _get_redis_client(redis_url: str):
    global _redis_client
    with _redis_cache_lock:
        if _redis_client is None:
            import redis.asyncio as redis_async

            _redis_client = redis_async.Redis.from_url(redis_url, decode_responses=True)
    return _redis_client


async def _get_cached_user_from_redis(cache_key: str) -> dict | None:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        client = _get_redis_client(redis_url)
        raw = await client.get(cache_key)
        if not raw:
            return None
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


async def _set_cached_user_in_redis(cache_key: str, payload: dict, ttl_seconds: int) -> None:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return
    try:
        client = _get_redis_client(redis_url)
        await client.set(cache_key, json.dumps(payload), ex=ttl_seconds)
    except Exception:
        return


async def _load_cached_user(cache_key: str) -> dict | None:
    redis_user = await _get_cached_user_from_redis(cache_key)
    if redis_user:
        return redis_user
    return _get_cached_user_from_memory(cache_key)


async def _store_cached_user(cache_key: str, payload: dict, ttl_seconds: int) -> None:
    _set_cached_user_in_memory(cache_key, payload, ttl_seconds)
    await _set_cached_user_in_redis(cache_key, payload, ttl_seconds)


async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    """
    Extracts user from Appwrite session JWT.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        token = credentials.credentials.strip()
        if not token:
            raise HTTPException(status_code=401, detail="Malformed Authorization header")
        cache_key = _token_cache_key(token)
        cached_user = await _load_cached_user(cache_key)
        if cached_user:
            return cached_user

        account = build_request_account(token)
        user = await asyncio.to_thread(account.get)
        auth_user = {
            "user_id": user["$id"],
            "email": user.get("email"),
            "name": user.get("name"),
        }
        await _store_cached_user(cache_key, auth_user, _cache_ttl_seconds())
        return auth_user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def ensure_user_scope(auth_user: dict, requested_user_id: str) -> None:
    auth_user_id = str((auth_user or {}).get("user_id", "")).strip()
    requested = str(requested_user_id or "").strip()
    if not requested:
        return
    if not auth_user_id:
        raise HTTPException(status_code=401, detail="Unauthorized user context")
    if requested != auth_user_id:
        raise HTTPException(status_code=403, detail="User scope mismatch")
