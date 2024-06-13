import json
import math
from typing import Dict, Any, List, Union

from core import Task
from core import tasks
from core.tasks import TaskConfig, TaskType
from models import base_models, utility_models
import bittensor as bt

MAX_SPEED_BONUS = 1.4  # Adjust this value as needed
BELOW_MEAN_EXPONENT = 3
CHARACTER_TO_TOKEN_CONVERSION = 4.0


def _calculate_speed_modifier(normalised_response_time: float, config: TaskConfig) -> float:
    """
    Calculates the speed modifier based on the normalised response time
    using sewed together gaussian distribution's
    """
    mean = config.mean
    variance = config.variance

    assert variance > 0

    bt.logging.warning(
        f"\nGetting the speed for task: {config.task} with normalised response time: {normalised_response_time}."
        f"\nMean: {mean}\nVariance: {variance}"
        f"\nBelow mean exponent: {BELOW_MEAN_EXPONENT}"
        f"\nMax speed bonus: {MAX_SPEED_BONUS}"
    )

    if normalised_response_time <= mean:
        # y = 1 + (M - 1) * (b - x)^c / b^c
        speed_modifier = 1 + (MAX_SPEED_BONUS - 1) * ((mean - normalised_response_time) ** BELOW_MEAN_EXPONENT) / (
            mean**BELOW_MEAN_EXPONENT
        )
        bt.logging.warning(f"Speed modifier in the if: {speed_modifier}.")
    else:
        # y = e^((b - x) * v)
        speed_modifier = math.exp((mean - normalised_response_time) * variance)
        bt.logging.warning(f"Speed modifier in the else: {speed_modifier}.")

    return speed_modifier


def _calculate_work_image(steps: int) -> float:
    """Returns the expected work for that image boi. Overhead is not needed in this calculation

    Volume for images is in terms of steps."""
    work = steps
    return work


def _calculate_work_text(character_count: int) -> float:
    """
    Returns the expected work for dem chars .

    Volume for text is tokems"""
    work = character_count / CHARACTER_TO_TOKEN_CONVERSION
    return work


def _calculate_work_clip(number_of_images: int) -> float:
    """
    Work for clip is just the number of images"""
    return number_of_images


def calculate_speed_modifier(
    task: Task, result: Union[utility_models.QueryResult, str], synapse: Dict[str, Any]
) -> float:
    config = tasks.get_task_config(task)

    response_time = result.response_time if isinstance(result, utility_models.QueryResult) else result["response_time"]
    raw_formatted_response = (
        result.formatted_response if isinstance(result, utility_models.QueryResult) else result["formatted_response"]
    )

    bt.logging.info(f"Response time: {response_time}, overhead: {config.overhead}")

    normalised_response_time = max(response_time - config.overhead, 0)

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
    config = tasks.get_task_config(task)

    raw_formatted_response = (
        result.formatted_response if isinstance(result, utility_models.QueryResult) else result["formatted_response"]
    )

    if config.task_type == TaskType.IMAGE:
        steps = synapse.get("steps", 1) if isinstance(synapse, dict) else synapse.steps
        return _calculate_work_image(steps)
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

        return _calculate_work_text(number_of_characters)
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
