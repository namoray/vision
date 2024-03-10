import httpx
import json
import os
from dotenv import load_dotenv
from fastapi import routing
from typing import  Dict, Any, Optional
import time
import asyncio

load_dotenv()


HEADERS = { 
    'Authorization': os.getenv('IMAGINE_SLASH_API_KEY'), 
    'Content-Type': 'application/json'
}
MAX_DURATION_TO_WAIT_FOR_IMAGE = 180

async def _create_image(prompt: str) -> Dict[str, Any]:

    async with httpx.AsyncClient() as client:
        data = {
            "prompt": prompt
        }
        response = await client.post(
            "https://api.slashimagine.pro/v3/imagine", headers=HEADERS, json=data
        )

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(f"Request failed with status {response.status_code}, error message: {response.text}")
        return {"result": "error"}


async def _get_image_from_task(task_id: str) -> Any:

    async with httpx.AsyncClient() as client:
        json={"taskId": task_id}
        print(json)
        response = await client.post(
            "https://api.slashimagine.pro/v3/result", headers=HEADERS, json=json
        )
    return response.json()

async def get_image(prompt: str) -> Optional[str]:
    create_image_response = await _create_image(prompt)
    task_id = create_image_response["task_id"]

    beginning_time = time.time()
    while time.time() - beginning_time < MAX_DURATION_TO_WAIT_FOR_IMAGE:
        task_response = await _get_image_from_task(task_id)
        status = task_response["status"]
        if status == "completed":
            return task_response["task_result"]["image_url"]
        else:
            # Want to wait so we don't get limited / blocked by the api
            await asyncio.sleep(1)
    return None
