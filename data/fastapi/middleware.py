import time

from fastapi import Request, Response


async def add_process_time_header(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.perf_counter() - start)
    return response
