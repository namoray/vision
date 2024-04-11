from typing import Dict, Any
from loguru import logger
import json
from typing import List
from models import utility_models
from models import base_models
from core.tasks import Tasks


def _calculate_speed_modifier(normalised_response_time: float, lower_bound: float, upper_bound: float) -> float:
    adjusted_response_time = max((normalised_response_time - lower_bound), 0)

    penalty = adjusted_response_time / (upper_bound - lower_bound)

    logger.info(
        f"Normalised response time: {normalised_response_time}; Lower bound: {lower_bound}; Upper bound: {upper_bound}; Penalty: {penalty}"
    )
    return max(1 - penalty**2, 0)


def _calculate_work_bonus_images(steps: int, overhead: float, lower_bound_for_seconds_per_step: float):
    """Returns a bonus based on the lower bound for computation"""

    bonus_flat = overhead + steps * lower_bound_for_seconds_per_step
    return bonus_flat**0.8


def _calculate_work_bonus_text(character_count: int, overhead: float, lower_bound_for_seconds_per_character: float):
    """Returns a bonus based on the lower bound for computation"""

    bonus_flat = overhead + character_count * lower_bound_for_seconds_per_character
    return bonus_flat**0.8


### SOTA ####


SOTA_OVERHEAD = 1
SOTA_LOWER_BOUND = 30
SOTA_MAX_ALLOWED_TIME = 120


async def speed_scoring_sota(result: utility_models.QueryResult, synapse: Dict[str, Any], task: str):
    speed_modifier = _calculate_speed_modifier(
        result.response_time - SOTA_OVERHEAD, SOTA_LOWER_BOUND, SOTA_MAX_ALLOWED_TIME
    )
    work_bonus = SOTA_LOWER_BOUND
    return work_bonus * speed_modifier


### CLIP  ####

CLIP_OVERHEAD = 1
CLIP_SUFFICIENTLY_QUICK_THRESHOLD_TIME = 0.5
CLIP_MAX_ALLOWED_TIME = 3


async def speed_scoring_clip(result: utility_models.QueryResult, synapse: Dict[str, Any], task: str) -> float:
    clip_result = base_models.ClipEmbeddingsOutgoing(**result.formatted_response)

    response_time = result.response_time

    number_of_clip_embeddings = len(clip_result.clip_embeddings)

    normalised_response_time = (
        max(response_time - CLIP_OVERHEAD + 0.1 * number_of_clip_embeddings, 0.1) / number_of_clip_embeddings
    )

    speed_modifier = _calculate_speed_modifier(
        normalised_response_time,
        CLIP_SUFFICIENTLY_QUICK_THRESHOLD_TIME,
        CLIP_MAX_ALLOWED_TIME,
    )
    work_bonus = number_of_clip_embeddings * CLIP_SUFFICIENTLY_QUICK_THRESHOLD_TIME + CLIP_OVERHEAD

    return speed_modifier * work_bonus


### Images  ####

T2I_OVERHEAD = 3
I2I_OVERHEAD = 3
UPSCALE_OVERHEAD = 5
AVATAR_OVERHEAD = 10
INPAINTING_OVERHEAD = 4


async def speed_scoring_images(result: utility_models.QueryResult, synapse: Dict[str, Any], task: str) -> float:
    response_time = result.response_time

    steps = synapse.get("steps", 1)

    if task == Tasks.proteus_text_to_image.value or task == Tasks.dreamshaper_text_to_image.value:
        response_time_without_overhead = max(response_time - T2I_OVERHEAD, 0.1)
        seconds_per_step = response_time_without_overhead / steps
        lower_bound_time = 0.5
        upper_thershold_time = 1.5

        speed_modifier = _calculate_speed_modifier(seconds_per_step, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_images(steps, T2I_OVERHEAD, lower_bound_time)

        return speed_modifier * work_bonus

    if task == Tasks.playground_text_to_image.value:
        response_time_without_overhead = max(response_time - T2I_OVERHEAD, 0.1)
        seconds_per_step = response_time_without_overhead / steps
        lower_bound_time = 0.2
        upper_thershold_time = 0.8

        speed_modifier = _calculate_speed_modifier(seconds_per_step, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_images(steps, T2I_OVERHEAD, lower_bound_time)

        return speed_modifier * work_bonus

    if task == Tasks.proteus_image_to_image.value or task == Tasks.dreamshaper_image_to_image.value:
        response_time_without_overhead = max(response_time - I2I_OVERHEAD, 0.1)
        seconds_per_step = response_time_without_overhead / steps
        lower_bound_time = 0.6
        upper_thershold_time = 1.6

        speed_modifier = _calculate_speed_modifier(seconds_per_step, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_images(steps, I2I_OVERHEAD, lower_bound_time)

        return speed_modifier * work_bonus

    if task == Tasks.playground_image_to_image.value:
        response_time_without_overhead = max(response_time - I2I_OVERHEAD, 0.1)
        seconds_per_step = response_time_without_overhead / steps
        lower_bound_time = 0.3
        upper_thershold_time = 0.9

        speed_modifier = _calculate_speed_modifier(seconds_per_step, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_images(steps, I2I_OVERHEAD, lower_bound_time)

        return speed_modifier * work_bonus

    if task == Tasks.jugger_inpainting.value:
        response_time_without_overhead = max(response_time - INPAINTING_OVERHEAD, 0.1)
        seconds_per_step = response_time_without_overhead / steps
        lower_bound_time = 0.5
        upper_thershold_time = 1.5

        speed_modifier = _calculate_speed_modifier(seconds_per_step, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_images(steps, INPAINTING_OVERHEAD, lower_bound_time)

        return speed_modifier * work_bonus

    if task == Tasks.avatar.value:
        response_time_without_overhead = max(response_time - AVATAR_OVERHEAD, 0.1)
        seconds_per_step = response_time_without_overhead / steps
        lower_bound_time = 0.5
        upper_thershold_time = 1.5

        speed_modifier = _calculate_speed_modifier(seconds_per_step, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_images(steps, AVATAR_OVERHEAD, lower_bound_time)
        return speed_modifier * work_bonus

    logger.error(f"Task {task} not found")
    return 1


### Chat

CHAT_OVERHEAD = 0.6


async def speed_scoring_chat(result: utility_models.QueryResult, synapse: Dict[str, Any], task: str) -> float:
    formatted_response = (
        json.loads(result.formatted_response)
        if isinstance(result.formatted_response, str)
        else result.formatted_response
    )
    miner_chat_responses: List[utility_models.MinerChatResponse] = [
        utility_models.MinerChatResponse(**r) for r in formatted_response
    ]

    all_text = "".join([mcr.text for mcr in miner_chat_responses])

    number_of_characters = len(all_text)

    if number_of_characters == 0:
        return 1

    response_time = result.response_time

    if task == Tasks.chat_bittensor_finetune.value:
        response_time_without_overhead = max(response_time - CHAT_OVERHEAD, 0.1)
        seconds_per_character = response_time_without_overhead / number_of_characters
        lower_bound_time = 1 / 120  # equivalent to ~ 30 tokens per second
        upper_thershold_time = 1 / 40  # equivalen to ~ 15 tokens per second

        speed_modifier = _calculate_speed_modifier(seconds_per_character, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_text(number_of_characters, CHAT_OVERHEAD, lower_bound_time)

        return speed_modifier * work_bonus

    if task == Tasks.chat_mixtral.value:
        response_time_without_overhead = max(response_time - CHAT_OVERHEAD, 0.1)
        seconds_per_character = response_time_without_overhead / number_of_characters
        lower_bound_time = 1 / 70  # equivalent to ~ 24 tokens per second
        upper_thershold_time = 1 / 30  # equivalen to ~ 15 tokens per second

        speed_modifier = _calculate_speed_modifier(seconds_per_character, lower_bound_time, upper_thershold_time)
        work_bonus = _calculate_work_bonus_text(number_of_characters, CHAT_OVERHEAD, lower_bound_time)

        return speed_modifier * work_bonus
