import base64
from io import BytesIO

from fastapi import APIRouter
from PIL import Image

from models import safety_models
from operation_logic import utils as operations_utils

router = APIRouter()


@router.post("/safety/check-image")
async def check_image(request: safety_models.CheckImageRequest) -> float:
    image_data = base64.b64decode(request.image_b64s[0])
    image = Image.open(BytesIO(image_data))

    return operations_utils.image_is_nsfw(image)
