import asyncio
import random
import threading
import httpx
from typing import Dict, Any
from config.validator_config import config as validator_config
from core import Task, tasks
import bittensor as bt

from models import base_models
from validation.proxy import validation_utils

SEED = "seed"
TEMPERATURE = "temperature"


class SyntheticDataManager:
    def __init__(self) -> None:
        self.task_to_stored_synthetic_data: Dict[Task, Dict[str, Any]] = {}

        thread = threading.Thread(target=self._start_async_loop, daemon=True)
        thread.start()

    def _start_async_loop(self):
        """Start the event loop and run the async tasks."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._continuously_fetch_synthetic_data_for_tasks())

    async def _continuously_fetch_synthetic_data_for_tasks(self) -> None:
        # Initial fetch be quick
        tasks_needing_synthetic_data = [task for task in tasks.Task if task not in self.task_to_stored_synthetic_data]
        while tasks_needing_synthetic_data:
            sync_tasks = []
            for task in tasks_needing_synthetic_data:
                sync_tasks.append(asyncio.create_task(self._update_synthetic_data_for_task(task)))

            await asyncio.gather(*sync_tasks)
            tasks_needing_synthetic_data = [
                task for task in tasks.Task if task not in self.task_to_stored_synthetic_data
            ]

        bt.logging.info("Got initial synthetic data!")

        while True:
            for task in tasks.Task:
                await self._update_synthetic_data_for_task(task)
            await asyncio.sleep(2)

    async def fetch_synthetic_data_for_task(self, task: Task) -> Dict[str, Any]:
        while task not in self.task_to_stored_synthetic_data:
            bt.logging.warning(f"Synthetic data not found for task {task} yet, waiting...")
            await asyncio.sleep(10)

        synth_data = self.task_to_stored_synthetic_data[task]
        task_config = tasks.get_task_config(task)
        if task_config.task_type == tasks.TaskType.IMAGE:
            synth_data[SEED] = random.randint(1, 1_000_000_000)
        elif task_config.task_type == tasks.TaskType.TEXT:
            synth_data[SEED] = random.randint(1, 1_000_000_000)
            synth_data[TEMPERATURE] = round(random.uniform(0, 1), 2)
        elif task_config.task_type == tasks.TaskType.CLIP:
            synth_model = base_models.ClipEmbeddingsIncoming(**synth_data)
            synth_model_altered = validation_utils.alter_clip_body(synth_model)
            synth_data = synth_model_altered.dict()

        return synth_data

    async def _update_synthetic_data_for_task(self, task: Task) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=7) as client:
                response = await client.post(
                    validator_config.external_server_url + "get-synthetic-data",
                    json={"task": task.value},
                )
                response.raise_for_status()  # raises an HTTPError if an unsuccessful status code was received
        except httpx.RequestError:
            # bt.logging.warning(f"Getting synthetic data error: {err.request.url!r}: {err}")
            return None
        except httpx.HTTPStatusError:
            # bt.logging.warning(
            #     f"Syntehtic data error; status code {err.response.status_code} while requesting {err.request.url!r}: {err}"
            # )
            return None

        try:
            response_json = response.json()
        except ValueError as e:
            bt.logging.error(f"Synthetic data Response contained invalid JSON: error :{e}")
            return None

        self.task_to_stored_synthetic_data[task] = response_json
