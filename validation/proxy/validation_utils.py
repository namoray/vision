import sys
import traceback
from typing import Optional
from typing import Union
from PIL import Image
from rich.console import Console
import bittensor as bt
import fastapi
from fastapi import HTTPException
import random
from core import utils as core_utils
from models import utility_models, base_models
import numpy as np

console = Console()


def log_task_exception(task):
    try:
        task.result()
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = "".join(tb_lines)
        bt.logging.error(f"Exception occurred: \n{tb_text}")


def handle_bad_result(result: Optional[Union[utility_models.QueryResult, str]]) -> None:
    if not isinstance(result, utility_models.QueryResult):
        message = "I'm sorry, no valid response was possible from the miners :/"
        if result is not None:
            message += f"\nThe error was: {result}"
        raise HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=message,
        )


def alter_image(
    pil_image: Image.Image,
) -> str:
    numpy_image = np.array(pil_image)
    for _ in range(3):
        rand_x, rand_y = (
            random.randint(0, pil_image.width - 1),
            random.randint(0, pil_image.height - 1),
        )

        for i in range(3):
            change = random.choice([-1, 1])
            numpy_image[rand_y, rand_x, i] = np.clip(numpy_image[rand_y, rand_x, i] + change, 0, 255)

    pil_image = Image.fromarray(numpy_image)

    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    new_image = core_utils.pil_to_base64(pil_image)
    return new_image


def alter_clip_body(
    body: base_models.ClipEmbeddingsIncoming,
) -> base_models.ClipEmbeddingsIncoming:
    if body.image_b64s is None:
        return body

    new_images = []
    for image in body.image_b64s:
        pil_image = core_utils.base64_to_pil(image)
        new_image = alter_image(pil_image)
        new_images.append(new_image)

    body.image_b64s = new_images
    return body
