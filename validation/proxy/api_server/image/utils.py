import fastapi
from pydantic import BaseModel
import bittensor as bt
from fastapi import HTTPException
from models import utility_models


class NSFWContentException(fastapi.HTTPException):
    def __init__(self, detail: str = "NSFW content detected"):
        super().__init__(status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


async def _do_nsfw_checks(formatted_response: BaseModel):
    if formatted_response.is_nsfw:
        raise NSFWContentException()
    
async def do_formatted_response_image_checks(formatted_response: BaseModel, result: utility_models.QueryResult):
    if formatted_response is None or formatted_response.image_b64 is None:
        # return a 500 internal server error and intenrally log it
        bt.logging.error(f"Received a None result for some reason; Result error message: {result.error_message}")
        raise HTTPException(status_code=500, detail=result.error_message)

    await _do_nsfw_checks(formatted_response)