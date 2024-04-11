import asyncio
import sys
import time
import traceback
from typing import Optional
from typing import Dict
from typing import Type
from typing import Union, Any
from rich.console import Console
from rich.table import Table
import bittensor as bt
import fastapi
import httpx
from config.validator_config import config as validator_config
from fastapi import HTTPException
from pydantic import BaseModel
import random
from core import constants as core_cst, utils as core_utils
from validation.proxy import constants as cst
from models import utility_models, base_models
import numpy as np
from PIL import Image
import sqlite3
from validation.proxy import speed_scoring_functions
import os
from core.tasks import Tasks

console = Console()


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
        response = httpx.get(base_url)
        return response.status_code == 200
    except httpx.RequestError:
        print(f"Health check failed for now - can't connect to {base_url}.")
        return False


def connect_to_external_server() -> str:
    hotkey_name = validator_config.hotkey_name

    servers = {
        core_cst.EXTERNAL_SERVER_ADDRESS_PARAM: validator_config.external_server_url,
    }

    # Check each server
    for name, url in servers.items():
        if url is None:
            raise Exception(f"{hotkey_name}.{name.upper()} not set in the config")

        retry_interval = 2
        while True:
            connected = health_check(url)
            if connected:
                bt.logging.info(f"Health check successful - connected to {name} at {url}.")
                break
            else:
                bt.logging.info(
                    f"{name} at url {url} not reachable just yet- it's probably still starting. Sleeping for {retry_interval} second(s) before retrying."
                )
                time.sleep(retry_interval)
                retry_interval += 5
                if retry_interval > 15:
                    retry_interval = 15


def alter_clip_body(
    body: base_models.ClipEmbeddingsIncoming,
) -> base_models.ClipEmbeddingsIncoming:
    if body.image_b64s is None:
        return body

    new_images = []
    for image in body.image_b64s:
        pil_image = core_utils.base64_to_pil(image)
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
        new_image = core_utils.pil_to_base64(pil_image)
        new_images.append(new_image)

    body.image_b64s = new_images
    return body


def store_and_print_scores(
    axon_scores: Dict[int, float],
    result1: utility_models.QueryResult,
    result2: utility_models.QueryResult,
    synapse: bt.Synapse,
    checked_with_server: bool,
    uid_to_uid_info: Dict[int, utility_models.UIDinfo],
):
    if os.path.isfile(core_cst.VALIDATOR_DB):
        conn = sqlite3.connect(core_cst.VALIDATOR_DB)
        cursor = conn.cursor()
    else:
        conn = None
        cursor = None

    timestamp = round(time.time(), 2)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("uid", style="dim")
    table.add_column("Response Time", style="dim")
    table.add_column("Score", style="dim")
    table.add_column("Synapse", justify="left", style="dim")
    table.add_column("Valid response", justify="left", style="dim")
    table.add_column("Quickest Response", justify="left", style="dim")
    table.add_column("Checked with server", justify="left", style="dim")

    for uid, score in axon_scores.items():
        if uid == result1.axon_uid:
            response_time = "N/A" if result1.response_time is None else str(round(result1.response_time, 2))
        elif uid == result2.axon_uid:
            response_time = "N/A" if result2.response_time is None else str(round(result2.response_time, 2))
        else:
            response_time = "N/A"

        valid_response = score > cst.FAILED_RESPONSE_SCORE
        quickest_response = score >= (1 + cst.BONUS_FOR_WINNING_MINER)

        if cursor is not None:
            hotkey = uid_to_uid_info[uid].hotkey

            row_data = (
                uid,
                hotkey,
                float(response_time) if response_time != "N/A" else None,
                round(score, 2),
                synapse.__class__.__name__,
                valid_response,
                quickest_response,
                checked_with_server,
                timestamp,
            )
            cursor.execute(
                "INSERT INTO scores (axon_uid, hotkey, response_time, score, synapse, valid_response, quickest_response, checked_with_server, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row_data,
            )
        table.add_row(
            str(uid),
            response_time,
            str(round(score, 2)),
            synapse.__class__.__name__,
            str(valid_response),
            str(quickest_response),
            str(checked_with_server),
        )

    console.print(table)

    if conn is not None:
        conn.commit()
        conn.close()


tasks_to_scoring_function = {
    Tasks.chat_bittensor_finetune.value: speed_scoring_functions.speed_scoring_chat,
    Tasks.chat_mixtral.value: speed_scoring_functions.speed_scoring_chat,
    Tasks.proteus_text_to_image.value: speed_scoring_functions.speed_scoring_images,
    Tasks.playground_text_to_image.value: speed_scoring_functions.speed_scoring_images,
    Tasks.dreamshaper_text_to_image.value: speed_scoring_functions.speed_scoring_images,
    Tasks.proteus_image_to_image.value: speed_scoring_functions.speed_scoring_images,
    Tasks.playground_image_to_image.value: speed_scoring_functions.speed_scoring_images,
    Tasks.dreamshaper_image_to_image.value: speed_scoring_functions.speed_scoring_images,
    Tasks.jugger_inpainting.value: speed_scoring_functions.speed_scoring_images,
    Tasks.avatar.value: speed_scoring_functions.speed_scoring_images,
    Tasks.clip_image_embeddings.value: speed_scoring_functions.speed_scoring_clip,
    Tasks.sota.value: speed_scoring_functions.speed_scoring_sota,
}


async def get_expected_score(result: utility_models.QueryResult, synapse: Dict[str, Any], task: str) -> float:
    expected_score =  await tasks_to_scoring_function[task](result, synapse, task)
    return max(expected_score, 1)
