from enum import Enum
import json
import math
from pydantic import BaseModel
from typing import Dict, Any, List, Union

from core import Task
from models import base_models, utility_models
import bittensor as bt


class TaskType(Enum):
    IMAGE = "image"
    TEXT = "text"
    CLIP = "clip"


class TaskConfig(BaseModel):
    task: Task
    overhead: float
    mean: float
    variance: float
    task_type: TaskType


TASK_CONFIGS = [
    TaskConfig(
        task=Task.proteus_text_to_image,
        overhead=3,
        mean=0.5,
        variance=1,
        task_type=TaskType.IMAGE,
    ),
    TaskConfig(
        task=Task.dreamshaper_text_to_image,
        overhead=3,
        mean=0.5,
        variance=6,
        task_type=TaskType.IMAGE,
    ),
    TaskConfig(
        task=Task.playground_text_to_image,
        overhead=3,
        mean=0.2,
        variance=6,
        task_type=TaskType.IMAGE,
    ),
    TaskConfig(
        task=Task.proteus_image_to_image,
        overhead=3,
        mean=0.6,
        variance=6,
        task_type=TaskType.IMAGE,
    ),
    TaskConfig(
        task=Task.dreamshaper_image_to_image,
        overhead=3,
        mean=0.6,
        variance=6,
        task_type=TaskType.IMAGE,
    ),
    TaskConfig(
        task=Task.playground_image_to_image,
        overhead=3,
        mean=0.3,
        variance=5,
        task_type=TaskType.IMAGE,
    ),
    TaskConfig(
        task=Task.jugger_inpainting,
        overhead=4,
        mean=0.5,
        variance=6,
        task_type=TaskType.IMAGE,
    ),
    TaskConfig(task=Task.avatar, overhead=5, mean=0.7, variance=6, task_type=TaskType.IMAGE),
    TaskConfig(
        task=Task.chat_mixtral,
        overhead=1,
        mean=1 / 80,
        variance=100,
        task_type=TaskType.TEXT,
    ),
    TaskConfig(
        task=Task.chat_llama_3,
        overhead=1,
        mean=1 / 80,
        variance=100,
        task_type=TaskType.TEXT,
    ),
    TaskConfig(
        task=Task.clip_image_embeddings,
        overhead=1,
        mean=0.5,
        variance=2,
        task_type=TaskType.CLIP,
    ),
]


MAX_SPEED_BONUS = 1.4  # Adjust this value as needed
BELOW_MEAN_EXPONENT = 3


def _calculate_speed_modifier(normalised_response_time: float, config: TaskConfig) -> float:
    """
    Calculates the speed modifier based on the normalised response time
    using sewed together gaussian distribution's
    """
    mean = config.mean
    variance = config.variance

    assert variance > 0

    if normalised_response_time <= mean:
        # y = (M - 1) * (b - x)^c / b^c + 1
        speed_modifier = 1 + (MAX_SPEED_BONUS - 1) * ((mean - normalised_response_time) ** BELOW_MEAN_EXPONENT) / (
            mean**BELOW_MEAN_EXPONENT
        )
    else:
        # y = e^((b - x) * v)
        speed_modifier = math.exp((mean - normalised_response_time) * variance)

    return speed_modifier


def _get_task_config(task: Task) -> TaskConfig:
    for config in TASK_CONFIGS:
        if config.task == task:
            return config
    raise ValueError(f"Task configuration for {task.value} not found")


def _calculate_work_image(steps: int, config: TaskConfig) -> float:
    """Returns the expected work for that image boi. Overhead is not needed in this calculation

    Volume for images is in terms of step generating seconds. Mean is in seconds per step"""
    work = steps * config.mean
    return work


def _calculate_work_text(character_count: int, config: TaskConfig) -> float:
    """
    Returns the expected work for dem chars .

    Volume for text is in terms of token generating seconds"""
    work = character_count * config.mean
    return work


def _calculate_work_clip(number_of_images: int) -> float:
    """
    Work for clip is just the number of images"""
    return number_of_images


def calculate_speed_modifier(
    task: Task, result: Union[utility_models.QueryResult, str], synapse: Dict[str, Any]
) -> float:
    config = _get_task_config(task)

    response_time = result.response_time if isinstance(result, utility_models.QueryResult) else result["response_time"]
    raw_formatted_response = (
        result.formatted_response if isinstance(result, utility_models.QueryResult) else result["formatted_response"]
    )

    normalised_response_time = response_time - config.overhead

    if config.task_type == TaskType.IMAGE:
        steps = synapse.get("steps", 1)
        time_per_step = normalised_response_time / steps
        return _calculate_speed_modifier(time_per_step, config)
    elif config.task_type == TaskType.TEXT:
        formatted_response = (
            json.loads(raw_formatted_response) if isinstance(raw_formatted_response, str) else raw_formatted_response
        )
        miner_chat_responses: List[utility_models.MinerChatResponse] = [
            utility_models.MinerChatResponse(**r) for r in formatted_response
        ]
        all_text = "".join([mcr.text for mcr in miner_chat_responses])
        number_of_characters = len(all_text)

        if number_of_characters == 0:
            return 0  # Doesn't matter what is returned here

        return _calculate_speed_modifier(normalised_response_time / number_of_characters, config)
    elif config.task_type == TaskType.CLIP:
        return _calculate_speed_modifier(normalised_response_time, config)
    else:
        raise ValueError(f"Task type {config.task_type} not found")


def calculate_work(
    task: Task, result: Union[utility_models.QueryResult, Dict[str, Any]], synapse: Union[Dict[str, Any], bt.Synapse]
) -> float:
    """Gets volume for the task that was executed"""
    config = _get_task_config(task)

    raw_formatted_response = (
        result.formatted_response if isinstance(result, utility_models.QueryResult) else result["formatted_response"]
    )

    if config.task_type == TaskType.IMAGE:
        steps = synapse.get("steps", 1) if isinstance(synapse, dict) else synapse.steps
        return _calculate_work_image(steps, config)
    elif config.task_type == TaskType.TEXT:
        formatted_response = (
            json.loads(raw_formatted_response) if isinstance(raw_formatted_response, str) else raw_formatted_response
        )
        miner_chat_responses: List[utility_models.MinerChatResponse] = [
            utility_models.MinerChatResponse(**r) for r in formatted_response
        ]
        all_text = "".join([mcr.text for mcr in miner_chat_responses])
        number_of_characters = len(all_text)

        if number_of_characters == 0:
            return 1

        return _calculate_work_text(number_of_characters, config)
    elif config.task_type == TaskType.CLIP:
        clip_result = (
            base_models.ClipEmbeddingsOutgoing(**json.loads(raw_formatted_response))
            if isinstance(raw_formatted_response, str)
            else base_models.ClipEmbeddingsOutgoing(**raw_formatted_response)
            if isinstance(raw_formatted_response, dict)
            else raw_formatted_response
        )
        return _calculate_work_clip(len(clip_result.clip_embeddings))
    else:
        raise ValueError(f"Task {task} not found for work bonus calculation")