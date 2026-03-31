from __future__ import annotations

import asyncio
import hashlib
import os
import time
from threading import Lock
from typing import Dict, Tuple


_memory_counters: Dict[str, Tuple[int, int]] = {}
_memory_lock = Lock()
_redis_clients_lock = Lock()
_redis_sync_client = None
_redis_async_client = None


def _counter_key(scope: str, subject: str, window_seconds: int) -> str:
    return f"rl:{scope}:{subject}:{int(time.time() // window_seconds)}"


def _in_memory_increment(key: str, *, window_seconds: int) -> int:
    now_window = int(time.time() // window_seconds)
    with _memory_lock:
        count, bucket = _memory_counters.get(key, (0, now_window))
        if bucket != now_window:
            count = 0
            bucket = now_window
        count += 1
        _memory_counters[key] = (count, bucket)
        return count


def _get_sync_redis_client(redis_url: str):
    global _redis_sync_client
    with _redis_clients_lock:
        if _redis_sync_client is None:
            import redis

            _redis_sync_client = redis.Redis.from_url(redis_url)
    return _redis_sync_client


def _get_async_redis_client(redis_url: str):
    global _redis_async_client
    with _redis_clients_lock:
        if _redis_async_client is None:
            import redis.asyncio as redis_async

            _redis_async_client = redis_async.Redis.from_url(redis_url)
    return _redis_async_client


def _increment_rate_counter_sync(scope: str, subject: str, *, limit: int, window_seconds: int = 60) -> bool:
    subject = str(subject or "").strip()
    if not subject:
        return False

    key = _counter_key(scope, subject, window_seconds)
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            client = _get_sync_redis_client(redis_url)
            value = client.incr(key)
            if value == 1:
                client.expire(key, window_seconds)
            return int(value) > int(limit)
        except Exception:
            pass

    value = _in_memory_increment(key, window_seconds=window_seconds)
    return int(value) > int(limit)


async def increment_rate_counter_async(scope: str, subject: str, *, limit: int, window_seconds: int = 60) -> bool:
    subject = str(subject or "").strip()
    if not subject:
        return False

    key = _counter_key(scope, subject, window_seconds)
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            client = _get_async_redis_client(redis_url)
            value = await client.incr(key)
            if value == 1:
                await client.expire(key, window_seconds)
            return int(value) > int(limit)
        except Exception:
            # Offload sync fallback so middleware never blocks the ASGI loop.
            return await asyncio.to_thread(
                _increment_rate_counter_sync,
                scope,
                subject,
                limit=limit,
                window_seconds=window_seconds,
            )

    value = _in_memory_increment(key, window_seconds=window_seconds)
    return int(value) > int(limit)


def increment_rate_counter(scope: str, subject: str, *, limit: int, window_seconds: int = 60) -> bool:
    return _increment_rate_counter_sync(scope, subject, limit=limit, window_seconds=window_seconds)


def token_subject(auth_header: str) -> str:
    raw = str(auth_header or "").strip()
    if not raw.lower().startswith("bearer "):
        return ""
    token = raw.split(" ", 1)[1].strip()
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
