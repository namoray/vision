import httpx
import json
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import routing
from typing import Optional, Dict, Any
import time
import asyncio
import bittensor as bt
load_dotenv()


HEADERS = {
    "X-API-KEY": os.getenv("GO_API_KEY"),
    "Content-Type": "application/json",
}

MAX_DURATION_TO_WAIT_FOR_IMAGE = 180
PROCESS_MODE = "turbo"


async def _create_image(prompt: str) -> Dict[ str, Any] | None:
    async with httpx.AsyncClient() as client:
        data = {
            "prompt": prompt,
            "process_mode": PROCESS_MODE,
            "aspect_ratio": "1:1",
        }
        response = await client.post(
            "https://api.midjourneyapi.xyz/mj/v2/imagine", headers=HEADERS, json=data
        )

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(
            f"Request failed with status {response.status_code}, error message: {response.text}"
        )


async def _get_image_from_task(task_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        json = {"task_id": task_id}
        response = await client.post(
            "https://api.midjourneyapi.xyz/mj/v2/fetch", headers=HEADERS, json=json
        )
    return response.json()


async def get_image(prompt: str) -> Optional[str]:

    bt.logging.info(f"Prompt: {prompt}")
    create_image_response = await _create_image(prompt)
    bt.logging.info("here 5")
    task_id = create_image_response["task_id"]

    beginning_time = time.time()
    while time.time() - beginning_time < MAX_DURATION_TO_WAIT_FOR_IMAGE:
        bt.logging.info("here 7")
        task_response = await _get_image_from_task(task_id)
        status = task_response["status"]
        if status == "finished":
            return task_response["task_result"]["image_url"]
        else:
            # Want to wait so we don't get limited / blocked by the api
            await asyncio.sleep(1)
    return None
