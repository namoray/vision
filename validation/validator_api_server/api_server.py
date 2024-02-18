
import fastapi
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import bittensor as bt
from core import constants as core_cst
from models import base_models, protocols, utility_models, request_models
from validation.validator_api_server import core_validator as cv
from validation.validator_api_server import validation_utils
from db import sql
from fastapi.responses import JSONResponse, Response
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS

from fastapi import Request


from core import utils
from core import resource_management
import yaml
from typing import Dict, Any
import uvicorn
import asyncio


app = FastAPI(debug=False)
core_validator = cv.CoreValidator()

class NSFWContentException(fastapi.HTTPException):
    def __init__(self, detail: str = "NSFW content detected"):
        super().__init__(status_code=fastapi.status.HTTP_206_PARTIAL_CONTENT, detail=detail)


async def _check_images_are_nsfw(formatted_response: BaseModel) -> bool:
    if formatted_response.image_b64s is None or len(formatted_response.image_b64s) == 0:
        return True

    url = core_validator.BASE_SAFETY_CHECKER_SERVER_URL + "safety/check-image"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"image_b64": formatted_response.image_b64s[0]},
        )
    return response.json()

async def _do_nsfw_checks(formatted_response: BaseModel):

    if formatted_response.image_b64s == [] and formatted_response.error_message is not None:
        if "nsfw" in formatted_response.error_message.lower():
            raise NSFWContentException()
    try:
        images_are_nsfw = await _check_images_are_nsfw(formatted_response)
    except Exception as e:
        bt.logging.error(f"Error when checking if images are nsfw: {e}. Will assume they are just to be safe")
        raise NSFWContentException()

    if images_are_nsfw:
        NSFWContentException()

async def _do_formatted_response_image_checks(formatted_response: BaseModel, result: utility_models.QueryResult):

    if formatted_response is None:
        # return a 500 internal server error and intenrally log it
        bt.logging.error(f"Received a None result for some reason; Result error message: {result.error_message}")
        raise HTTPException(status_code=500, detail=result.error_message)

    await _do_nsfw_checks(formatted_response)

@app.post("/text-to-image")
async def text_to_image(
    body: request_models.TextToImageRequest,
) -> request_models.TextToImageResponse:
    synapse = validation_utils.get_synapse_from_body(
        body=body,
        synapse_model=protocols.TextToImage,
    )

    bt.logging.info(f"Text to image synapse: {synapse}")

    result: utility_models.QueryResult = await core_validator.execute_query(
        synapse, outgoing_model=base_models.TextToImageOutgoing
    )
    validation_utils.handle_bad_result(result)

    formatted_response: base_models.TextToImageOutgoing = result.formatted_response

    await _do_formatted_response_image_checks(formatted_response, result)
    return request_models.TextToImageResponse(image_b64s=formatted_response.image_b64s)


@app.post("/image-to-image")
async def image_to_image(
    body: request_models.ImageToImageRequest,
) -> request_models.ImageToImageResponse:
    synapse = validation_utils.get_synapse_from_body(
        body=body,
        synapse_model=protocols.ImageToImage,
    )

    result = await core_validator.execute_query(synapse, outgoing_model=base_models.ImageToImageOutgoing)
    validation_utils.handle_bad_result(result)

    formatted_response: base_models.ImageToImageOutgoing = result.formatted_response


    await _do_formatted_response_image_checks(formatted_response, result)
    return request_models.ImageToImageResponse(image_b64s=formatted_response.image_b64s)


@app.post("/inpaint")
async def inpaint(
    body: request_models.InpaintRequest,
) -> request_models.InpaintResponse:
    synapse = validation_utils.get_synapse_from_body(
        body=body,
        synapse_model=protocols.Inpaint,
    )

    result = await core_validator.execute_query(synapse, outgoing_model=base_models.InpaintOutgoing)
    validation_utils.handle_bad_result(result)

    formatted_response: base_models.InpaintOutgoing = result.formatted_response


    await _do_formatted_response_image_checks(formatted_response, result)

    return request_models.InpaintResponse(image_b64s=formatted_response.image_b64s)


@app.post("/scribble")
async def scribble(
    body: request_models.ScribbleRequest,
) -> request_models.ScribbleResponse:
    synapse = validation_utils.get_synapse_from_body(
        body=body,
        synapse_model=protocols.Scribble,
    )

    result = await core_validator.execute_query(synapse, outgoing_model=base_models.ScribbleOutgoing)
    validation_utils.handle_bad_result(result)

    formatted_response: base_models.ScribbleOutgoing = result.formatted_response


    await _do_formatted_response_image_checks(formatted_response, result)

    return request_models.ScribbleResponse(image_b64s=formatted_response.image_b64s)


@app.post("/upscale")
async def upscale(
    body: request_models.UpscaleRequest,
) -> request_models.UpscaleResponse:
    synapse = validation_utils.get_synapse_from_body(
        body=body,
        synapse_model=protocols.Upscale,
    )

    result = await core_validator.execute_query(synapse, outgoing_model=base_models.UpscaleOutgoing)
    validation_utils.handle_bad_result(result)

    formatted_response: base_models.UpscaleOutgoing = result.formatted_response


    await _do_formatted_response_image_checks(formatted_response, result)

    return request_models.UpscaleResponse(image_b64s=formatted_response.image_b64s)


@app.post("/clip-embeddings")
async def clip_embeddings(
    body: request_models.ClipEmbeddingsRequest,
) -> request_models.ClipEmbeddingsResponse:
    altered_clip_body = validation_utils.alter_clip_body(body)
    synapse = validation_utils.get_synapse_from_body(
        body=altered_clip_body,
        synapse_model=protocols.ClipEmbeddings,
    )

    result = await core_validator.execute_query(synapse, outgoing_model=base_models.ClipEmbeddingsOutgoing)
    if result is None:
        raise HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail="I'm sorry, no valid response was possible from the miners :/",
        )

    formatted_response: base_models.ClipEmbeddingsOutgoing = result.formatted_response

    return request_models.ClipEmbeddingsResponse(
        image_embeddings=formatted_response.image_embeddings,
    )



def _get_api_key(request: Request):
    return request.headers.get("X-API-KEY")

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
            status_code=HTTP_400_BAD_REQUEST,
            content={"detail": "API key is missing"},
        )

    with sql.get_db_connection() as conn:
        api_key_info = sql.get_api_key_info(conn, api_key)

    if api_key_info is None:
        return JSONResponse(status_code=HTTP_401_UNAUTHORIZED, content={"detail": "Invalid API key"})

    bt.logging.warning(f"api key info keys: {api_key_info.keys()}")

    endpoint = request.url.path.split("/")[-1]
    credits_required = ENDPOINT_TO_CREDITS_USED.get(endpoint, 1)

    # Now check credits
    if api_key_info[sql.BALANCE] is not None and api_key_info[sql.BALANCE] <= credits_required:
        return JSONResponse(status_code=HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Insufficient credits - sorry!"})

    # Now check rate limiting
    with sql.get_db_connection() as conn:
        rate_limit_exceeded = sql.rate_limit_exceeded(conn, api_key_info)
        if rate_limit_exceeded:
            return JSONResponse(status_code=HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Rate limit exceeded - sorry!"})

    response: Response = await call_next(request)

    bt.logging.debug(f"response: {response}")
    if response.status_code == 200:
        with sql.get_db_connection() as conn:
            sql.update_requests_and_credits(conn, api_key_info, credits_required)
            sql.log_request(conn, api_key_info, request.url.path, credits_required)
            conn.commit()
    return response



async def main():

    yaml_config: Dict[str, Any] = yaml.safe_load(open(core_cst.CONFIG_FILEPATH))

    validator_hotkey_name = utils.get_validator_hotkey_name_from_config(yaml_config)
    if validator_hotkey_name is None:
        raise ValueError("Please set up the config for a validator!")
    resource_management.set_hotkey_name(validator_hotkey_name)

    core_validator.start_continuous_tasks()

    port = yaml_config.get(validator_hotkey_name, {}).get(core_cst.API_SERVER_PORT_PARAM)

    if port is not None:
        uvicorn.run(app, host="0.0.0.0", port=int(port), loop="asyncio")
    else:
        while True:
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())


# @app.post("/segment")
# async def segment(
#     body: request_models.SegementRequest,
#
# ) -> base_models.SegmentOutgoing:
#     synapse = validation_utils.get_synapse_from_body(
#         body=body,
#         synapse_model=protocols.Segment,
#     )

#     result = await core_validator.execute_query(synapse, outgoing_model=base_models.SegmentOutgoing)
#     if result is None:
#         raise HTTPException(
#             status_code=fastapi.status.HTTP_400_BAD_REQUEST,
#             detail="I'm sorry, no valid response was possible from the miners :/",
#         )

#     formatted_response: base_models.SegmentOutgoing = result.formatted_response

#     return formatted_response
