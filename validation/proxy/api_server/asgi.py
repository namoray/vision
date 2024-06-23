from fastapi import FastAPI
from fastapi import Request
import uvicorn
import asyncio
from config.validator_config import config as validator_config
from validation.proxy.api_server.image.endpoints import router as image_router
from validation.proxy.api_server.text.endpoints import router as text_router
from validation.core_validator import core_validator
from validation.proxy import sql
from fastapi.responses import JSONResponse
from starlette import status
from fastapi import Response
from validation.db.db_management import db_manager

app = FastAPI(debug=False)


app.include_router(image_router)
app.include_router(text_router)


async def main():
    await db_manager.initialize()
    core_validator.start_continuous_tasks()

    port = validator_config.api_server_port

    if port:
        uvicorn.run(app, host="0.0.0.0", port=int(port), loop="asyncio")
    else:
        while True:
            await asyncio.sleep(10)


def _get_api_key(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    else:
        return auth_header


ENDPOINT_TO_CREDITS_USED = {
    "clip-embeddings": 0.2,
    "text-to-image": 1,
    "image-to-image": 1,
    "inpaint": 1,
    "scribble": 1,
    "upscale": 1,
}


@app.middleware("http")
async def api_key_validator(request, call_next):
    if request.url.path in ["/docs", "/openapi.json", "/favicon.ico", "/redoc"]:
        return await call_next(request)

    api_key = _get_api_key(request)
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "API key is missing"},
        )

    with sql.get_db_connection() as conn:
        api_key_info = sql.get_api_key_info(conn, api_key)

    if api_key_info is None:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Invalid API key"})
    endpoint = request.url.path.split("/")[-1]
    credits_required = ENDPOINT_TO_CREDITS_USED.get(endpoint, 1)

    # Now check credits
    if api_key_info[sql.BALANCE] is not None and api_key_info[sql.BALANCE] <= credits_required:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Insufficient credits - sorry!"}
        )

    # Now check rate limiting
    with sql.get_db_connection() as conn:
        rate_limit_exceeded = sql.rate_limit_exceeded(conn, api_key_info)
        if rate_limit_exceeded:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Rate limit exceeded - sorry!"}
            )

    response: Response = await call_next(request)

    if response.status_code == 200:
        with sql.get_db_connection() as conn:
            sql.update_requests_and_credits(conn, api_key_info, credits_required)
            sql.log_request(conn, api_key_info, request.url.path, credits_required)
            conn.commit()
    return response


if __name__ == "__main__":
    asyncio.run(main())
