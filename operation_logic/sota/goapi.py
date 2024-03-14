import httpx
import json
from typing import Optional, Dict, Any
import time
import asyncio
import bittensor as bt


MAX_DURATION_TO_WAIT_FOR_IMAGE = 180
PROCESS_MODE = "turbo"


async def _create_image(prompt: str, sota_key: str) -> Dict[ str, Any] | None:

    bt.logging.info(f"Prompt: {prompt}")
    bt.logging.info(f"Process mode: {PROCESS_MODE}")
    bt.logging.info(f"API key: {sota_key}")
    async with httpx.AsyncClient() as client:
        data = {
            "prompt": prompt,
            "process_mode": PROCESS_MODE,
            "aspect_ratio": "1:1",
        }
        headers = {
        "X-API-KEY": sota_key,
        "Content-Type": "application/json",
        }
        bt.logging.info(f"data: {data}, headers: {headers}")
        response = await client.post(
            "https://api.midjourneyapi.xyz/mj/v2/imagine", headers=headers, json=data
        )

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(
            f"Request failed with status {response.status_code}, error message: {response.text}"
        )


async def _get_image_from_task(task_id: str, sota_key: str) -> Dict[str, Any]:
    
    async with httpx.AsyncClient() as client:
        json = {"task_id": task_id}
        headers = {
            "X-API-KEY": sota_key,
            "Content-Type": "application/json",
        }
        response = await client.post(
            "https://api.midjourneyapi.xyz/mj/v2/fetch", headers=headers, json=json
        )
    return response.json()


async def get_image(prompt: str, sota_key: str) -> Optional[str]:

    bt.logging.info(f"Prompt: {prompt}")

    create_image_response = await _create_image(prompt, sota_key)
    task_id = create_image_response["task_id"]

    beginning_time = time.time()
    while time.time() - beginning_time < MAX_DURATION_TO_WAIT_FOR_IMAGE:
        task_response = await _get_image_from_task(task_id, sota_key)
        status = task_response["status"]
        if status == "finished":
            return task_response["task_result"]["image_url"]
        else:
            # Want to wait so we don't get limited / blocked by the api
            await asyncio.sleep(1)
    return None
