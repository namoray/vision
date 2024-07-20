# Schema for the db
from typing import Dict, List

from core import Task
from models import utility_models
from validation.db.db_management import db_manager
from validation.models import PeriodScore
from validation.models import axon_uid
from validation.models import RewardData

PERIOD_SCORE_TIME_DECAYING_FACTOR = 0.5


async def _get_reward_datas(miner_hotkey: str, task: Task) -> List[RewardData]:
    reward_datas = await db_manager.fetch_recent_most_rewards_for_uid(task, miner_hotkey)
    return reward_datas


async def _get_period_scores(miner_hotkey: str, task: Task) -> List[PeriodScore]:
    period_scores = await db_manager.fetch_hotkey_scores_for_task(task, miner_hotkey)
    return period_scores


async def _calculate_combined_quality_score(miner_hotkey: str, task: Task) -> float:
    reward_datas = await _get_reward_datas(miner_hotkey, task)
    combined_quality_scores = [
        reward_data.quality_score * reward_data.speed_scoring_factor for reward_data in reward_datas
    ]
    if not combined_quality_scores:
        return 0
    return sum(combined_quality_scores) / len(combined_quality_scores)


async def _calculate_normalised_period_score(miner_hotkey: str, task: Task) -> float:
    period_scores = await _get_period_scores(miner_hotkey, task)
    all_period_scores = [ps for ps in period_scores if ps.period_score is not None]
    normalised_period_scores = _normalise_period_scores(all_period_scores)
    return normalised_period_scores


def _normalise_period_scores(period_scores: List[PeriodScore]) -> float:
    if len(period_scores) == 0:
        return 0

    sum_of_volumes = sum(ps.consumed_volume for ps in period_scores)
    if sum_of_volumes == 0:
        return 0

    total_score = 0
    total_weight = 0
    for i, score in enumerate(period_scores):
        volume_weight = score.consumed_volume / sum_of_volumes
        time_weight = (1 - PERIOD_SCORE_TIME_DECAYING_FACTOR) ** i
        combined_weight = volume_weight * time_weight
        total_score += score.period_score * combined_weight
        total_weight += combined_weight

    if total_weight == 0:
        return 0
    else:
        # Make sure UID's which are new don't instantly jump to the top
        penalty = max(0, 1 - len(period_scores) / 8) / 4
        return total_score / total_weight * (1 - penalty)


def _calculate_hotkey_effective_volume_for_task(
    combined_quality_score: float, normalised_period_score: float, volume: float
) -> float:
    return combined_quality_score * normalised_period_score * volume


def _non_linear_score_transformation(effective_volume_for_task: float) -> float:
    """
    This non-linear transformation helps reward clear outstanding UIDs, encouraging top behaviour
    """
    return effective_volume_for_task**3


def apply_non_linear_transformation_and_renormalise(linear_normalised_scores: Dict[str, float]) -> Dict[str, float]:
    transformed_scores = {
        hotkey: _non_linear_score_transformation(score) for hotkey, score in linear_normalised_scores.items()
    }

    sum_transformed_scores = sum(transformed_scores.values())

    return {hotkey: score / sum_transformed_scores for hotkey, score in transformed_scores.items()}


async def calculate_scores_for_settings_weights(
    capacities_for_tasks: Dict[Task, Dict[axon_uid, float]],
    uid_to_uid_info: Dict[axon_uid, utility_models.UIDinfo],
    task_weights: Dict[Task, float],
) -> Dict[str, float]:
    total_hotkey_scores: Dict[str, float] = {}

    for task in Task:
        if task not in task_weights:
            continue
        task_weight = task_weights[task]
        hotkey_to_effective_volumes: Dict[str, float] = {}
        capacities = capacities_for_tasks[task]

        for uid, volume in capacities.items():
            miner_hotkey = uid_to_uid_info[uid].hotkey
            combined_quality_score = await _calculate_combined_quality_score(miner_hotkey, task)
            normalised_period_score = await _calculate_normalised_period_score(miner_hotkey, task)
            effective_volume_for_task = _calculate_hotkey_effective_volume_for_task(
                combined_quality_score, normalised_period_score, volume
            )
            hotkey_to_effective_volumes[miner_hotkey] = effective_volume_for_task

        sum_of_effective_volumes = sum(hotkey_to_effective_volumes.values())
        if sum_of_effective_volumes == 0:
            continue
        normalised_scores_for_task = {
            hotkey: effective_volume / sum_of_effective_volumes
            for hotkey, effective_volume in hotkey_to_effective_volumes.items()
        }

        normalised_scores_for_task = apply_non_linear_transformation_and_renormalise(normalised_scores_for_task)
        for hotkey in normalised_scores_for_task:
            total_hotkey_scores[hotkey] = (
                total_hotkey_scores.get(hotkey, 0) + normalised_scores_for_task[hotkey] * task_weight
            )

    return total_hotkey_scores
