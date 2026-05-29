import random
import time
import httpx


def with_retry(fn, max_retries=3, base_delay=1.0):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
    raise last_exception


def send_callback(callback_url: str, callback_secret: str, payload: dict):
    if not callback_url:
        return
    headers = {}
    if callback_secret:
        headers["X-Modal-Secret"] = callback_secret
    with_retry(lambda: httpx.post(
        callback_url,
        json=payload,
        headers=headers,
        timeout=30
    ))
