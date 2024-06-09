import asyncio
import random
import threading
import httpx
from typing import Dict, Any
from config.validator_config import config as validator_config
from core import Task, tasks
import bittensor as bt

SEED = "seed"


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
        bt.logging.error(f"here!")
        # Initial fetch be quick
        initial_sync_tasks = []
        for task in tasks.Task:
            initial_sync_tasks.append(asyncio.create_task(self._update_synthetic_data_for_task(task)))

        await asyncio.gather(*initial_sync_tasks)

        while True:
            for task in tasks.Task:
                await self._update_synthetic_data_for_task(task)
            await asyncio.sleep(2)

    async def fetch_synthetic_data_for_task(self, task: Task) -> Dict[str, Any]:
        while task not in self.task_to_stored_synthetic_data:
            bt.logging.warning(f"Synthetic data not found for task {task} yet, waiting...")
            await asyncio.sleep(10)

        synth_data = self.task_to_stored_synthetic_data[task]
        bt.logging.info(f"Synthetic data found for task {task}: {synth_data}")
        task_config = tasks.get_task_config(task)
        if task_config.task_type in tasks.TaskType.IMAGE:
            synth_data[SEED] = random.randint(1, 1_000_000_000)
        elif task_config.task_type in tasks.TaskType.TEXT:
            ...
        elif task_config.task_type in tasks.TaskType.CLIP:
            ...

        return synth_data

    async def _update_synthetic_data_for_task(self, task: Task) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=7) as client:
                response = await client.post(
                    validator_config.external_server_url + "get-synthetic-data",
                    json={"task": task.value},
                )
                response.raise_for_status()  # raises an HTTPError if an unsuccessful status code was received
        except httpx.RequestError as err:
            bt.logging.warning(f"Getting synthetic data error: {err.request.url!r}: {err}")
            return None
        except httpx.HTTPStatusError as err:
            bt.logging.warning(
                f"Syntehtic data error; status code {err.response.status_code} while requesting {err.request.url!r}: {err}"
            )
            return None

        try:
            response_json = response.json()
        except ValueError as e:
            bt.logging.error(f"Synthetic data Response contained invalid JSON: error :{e}")
            return None

        bt.logging.info(f"Got synthetic data for task {task}: {response_json}")
        self.task_to_stored_synthetic_data[task] = response_json
