import asyncio
import base64
import random
import string
import threading
import httpx
from typing import Dict, Any
from config.validator_config import config as validator_config
from core import Task, tasks
import bittensor as bt
from core import dataclasses as dc
from models import base_models
from validation.proxy import validation_utils
from core import utils as core_utils
from PIL.Image import Image

SEED = "seed"
TEMPERATURE = "temperature"
TEXT_PROMPTS = "text_prompts"


def load_postie_to_pil(image_path: str) -> Image:
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    pil_image = core_utils.base64_to_pil(base64_string)
    return pil_image


my_boy_postie = load_postie_to_pil("validation/synthetic_data/postie.png")


def _get_random_letters(length: int) -> str:
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(length))


def _get_random_avatar_text_prompt() -> dc.TextPrompt:
    nouns = ['king', 'man', 'woman', 'joker', 'queen', 'child', 'doctor', 'teacher', 'soldier', 'merchant']  # fmt: off
    locations = ['forest', 'castle', 'city', 'village', 'desert', 'oceanside', 'mountain', 'garden', 'library', 'market']  # fmt: off
    looks = ['happy', 'sad', 'angry', 'worried', 'curious', 'lost', 'busy', 'relaxed', 'fearful', 'thoughtful']  # fmt: off
    actions = ['running', 'walking', 'reading', 'talking', 'sleeping', 'dancing', 'working', 'playing', 'watching', 'singing']  # fmt: off
    times = ['in the morning', 'at noon', 'in the afternoon', 'in the evening', 'at night', 'at midnight', 'at dawn', 'at dusk', 'during a storm', 'during a festival']  # fmt: off

    noun = random.choice(nouns)
    location = random.choice(locations)
    look = random.choice(looks)
    action = random.choice(actions)
    time = random.choice(times)

    text = f"{noun} in a {location}, looking {look}, {action} {time}"
    return dc.TextPrompt(text=text, weight=1.0)


def _my_boy_postie() -> str:
    b64_postie_altered = validation_utils.alter_image(my_boy_postie)
    return b64_postie_altered


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

        while True:
            for task in tasks.Task:
                await self._update_synthetic_data_for_task(task)
                await asyncio.sleep(3)

    async def fetch_synthetic_data_for_task(self, task: Task) -> Dict[str, Any]:
        while task not in self.task_to_stored_synthetic_data:
            bt.logging.warning(f"Synthetic data not found for task {task} yet, waiting...")
            await asyncio.sleep(10)

        synth_data = self.task_to_stored_synthetic_data[task]
        task_config = tasks.get_task_config(task)
        if task_config.task_type == tasks.TaskType.IMAGE:
            synth_data[SEED] = random.randint(1, 1_000_000_000)
            text_prompts = synth_data[TEXT_PROMPTS]
            text = text_prompts[0]["text"]
            new_text = text + _get_random_letters(4)
            synth_data[TEXT_PROMPTS][0]["text"] = new_text
        elif task_config.task_type == tasks.TaskType.TEXT:
            synth_data[SEED] = random.randint(1, 1_000_000_000)
            synth_data[TEMPERATURE] = round(random.uniform(0, 1), 2)
        elif task_config.task_type == tasks.TaskType.CLIP:
            synth_model = base_models.ClipEmbeddingsIncoming(**synth_data)
            synth_model_altered = validation_utils.alter_clip_body(synth_model)
            synth_data = synth_model_altered.dict()

        return synth_data

    async def _update_synthetic_data_for_task(self, task: Task) -> Dict[str, Any]:
        if task == Task.avatar:
            synthetic_data = base_models.AvatarIncoming(
                seed=random.randint(1, 1_000_000_000),
                text_prompts=[_get_random_avatar_text_prompt()],
                height=1280,
                width=1280,
                steps=15,
                control_strength=0.5,
                ipadapter_strength=0.5,
                init_image=_my_boy_postie(),
            ).dict()
            self.task_to_stored_synthetic_data[task] = synthetic_data
        else:
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
