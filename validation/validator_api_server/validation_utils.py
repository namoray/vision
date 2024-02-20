import asyncio
import sys
import time
import traceback
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import bittensor as bt
import fastapi
import httpx
import yaml
from fastapi import HTTPException
from pydantic import BaseModel
import random
from core import constants as core_cst, utils as core_utils
from models import utility_models, base_models
import numpy as np
from PIL import Image

def log_task_exception(task):
    try:
        task.result()
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = "".join(tb_lines)
        bt.logging.error(f"Exception occurred: \n{tb_text}")
        asyncio.get_event_loop().stop()


def get_synapse_from_body(
    body: BaseModel,
    synapse_model: Type[bt.Synapse],
) -> bt.Synapse:
    body_dict = body.dict()
    body_dict["seed"] = random.randint(1, core_cst.LARGEST_SEED)
    synapse = synapse_model(**body_dict)
    return synapse


def handle_bad_result(result: Optional[Union[utility_models.QueryResult, str]]) -> None:
    if not isinstance(result, utility_models.QueryResult):
        message = "I'm sorry, no valid response was possible from the miners :/"
        if result is not None:
            message += f"\nThe error was: {result}"
        raise HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=message,
        )


def health_check(base_url):
    try:
        response = httpx.get(base_url + "health")
        return response.status_code == 200
    except httpx.RequestError:
        bt.logging.warning(f"Health check failed - can't connect to {base_url}.")
        return False


def connect_to_checking_servers(config) -> Tuple[str, str]:

    hotkey_name = config.wallet.hotkey
    config = yaml.safe_load(open("config.yaml"))

    servers = {
        "checking_server_url": config.get(hotkey_name).get("CHECKING_SERVER_ADDRESS", None),
        "safety_checker_server_url": config.get(hotkey_name).get("SAFETY_CHECKER_SERVER_ADDRESS", None),
    }

    # Check each server
    for name, url in servers.items():
        if url is None:
            raise Exception(f"{hotkey_name}.{name.upper()} not set in config.yaml")

        retry_interval = 6
        while True:
            connected = health_check(url)
            if connected:
                bt.logging.info(f"Health check successful - connected to {name} at {url}.")
                break
            else:
                bt.logging.info(f"{name} not reachable just yet- it's probably still starting. Sleeping for {retry_interval} second(s) before retrying.")
                time.sleep(retry_interval)
                retry_interval += 0.5
                if retry_interval > 10:
                    retry_interval = 10

    return servers["checking_server_url"], servers["safety_checker_server_url"]

def alter_clip_body(body: base_models.ClipEmbeddingsIncoming) -> base_models.ClipEmbeddingsIncoming:
    if body.image_b64s is None:
        return body

    new_images = []
    for image in body.image_b64s:
        pil_image = core_utils.base64_to_pil(image)
        numpy_image = np.array(pil_image)
        for _ in range(3):

            rand_x, rand_y = random.randint(0, pil_image.width-1), random.randint(0, pil_image.height-1)

            for i in range(3):
                change = random.choice([-1, 1])
                numpy_image[rand_y, rand_x, i] = np.clip(numpy_image[rand_y, rand_x, i] + change, 0, 255)

        pil_image = Image.fromarray(numpy_image)
        new_image = core_utils.pil_to_base64(pil_image)
        new_images.append(new_image)

    body.image_b64s = new_images
    return body
