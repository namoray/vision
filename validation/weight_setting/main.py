# Schema for the db
import asyncio
import time
from typing import Dict, List

import bittensor as bt
import torch
from core import Task
from models import utility_models
from validation.db.db_management import db_manager
from validation.models import PeriodScore, RewardData
from validation.models import axon_uid


VERSION_KEY = 40_000

# If 1 then only take into account the most recent. If zero then they are all equal
PERIOD_SCORE_TIME_DECAYING_FACTOR = 0.5


class WeightSetter:
    def __init__(self, subtensor: bt.subtensor, config: bt.config) -> None:
        self.subtensor = subtensor
        self.config = config

    async def start_weight_setting_process(
        self,
        metagraph: bt.metagraph,
        wallet: bt.wallet,
        netuid: int,
        capacities_for_tasks: Dict[Task, Dict[axon_uid, float]],
        uid_to_uid_info: Dict[axon_uid, utility_models.UIDinfo],
        task_importances: Dict[Task, float],
    ) -> None:
        total_hotkey_scores = self._calculate_scores_for_settings_weights(
            capacities_for_tasks, uid_to_uid_info, task_importances
        )
        await asyncio.to_thread(self._set_weights, metagraph, wallet, netuid, total_hotkey_scores, uid_to_uid_info)

    def _set_weights(
        self,
        metagraph: bt.metagraph,
        wallet: bt.wallet,
        netuid: int,
        total_hotkey_scores: Dict[str, float],
        uid_to_uid_info: Dict[axon_uid, utility_models.UIDinfo],
    ) -> None:
        hotkey_to_uid = {uid_info.hotkey: uid_info.uid for uid_info in uid_to_uid_info.values()}
        weights_tensor = torch.zeros_like(metagraph.S, dtype=torch.float32)
        for hotkey, score in total_hotkey_scores.items():
            uid = hotkey_to_uid[hotkey]
            weights_tensor[uid] = score

        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=metagraph.uids.to("cpu"),
            weights=weights_tensor.to("cpu"),
            netuid=netuid,
            subtensor=self.subtensor,
            metagraph=metagraph,
        )

        bt.logging.info(f"Weights set to: {processed_weights} for uids: {processed_weight_uids}")

        NUM_TIMES_TO_SET_WEIGHTS = 3
        # The reason we do this is because wait_for_inclusion & wait_for_finalization
        # Cause the whole API server to crash.
        # So we have no choice but to set weights
        bt.logging.info(f"\n\nSetting weights {NUM_TIMES_TO_SET_WEIGHTS} times without inclusion or finalization\n\n")
        for i in range(NUM_TIMES_TO_SET_WEIGHTS):
            bt.logging.info(f"Setting weights, iteration number: {i+1}")
            success = self.subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=processed_weight_uids,
                weights=processed_weights,
                version_key=VERSION_KEY,
                wait_for_finalization=False,
                wait_for_inclusion=False,
            )

            if success:
                bt.logging.info("✅ Done setting weights!")
            time.sleep(30)

    @staticmethod
    def _calculate_scores_for_settings_weights(
        capacities_for_tasks: Dict[Task, Dict[axon_uid, float]],
        uid_to_uid_info: Dict[axon_uid, utility_models.UIDinfo],
        task_importances: Dict[Task, float],
    ):
        total_hotkey_scores: Dict[str, float] = {}
        for task in Task:
            hotkey_to_overall_scores: Dict[str, float] = {}
            capacities = capacities_for_tasks[task]
            if task not in task_importances:
                continue
            importance = task_importances[task]

            for uid in capacities:
                miner_hotkey = uid_to_uid_info[uid].hotkey
                volume = capacities[uid]
                reward_datas: List[RewardData] = db_manager.fetch_recent_most_rewards_for_uid(task, miner_hotkey)
                combined_quality_scores = []
                for reward_data in reward_datas:
                    combined_quality_scores.append(reward_data.quality_score * reward_data.speed_scoring_factor)

                combined_quality_score = (
                    0
                    if len(combined_quality_scores) == 0
                    else sum(combined_quality_scores) / len(combined_quality_scores)
                )

                period_scores = db_manager.fetch_hotkey_scores_for_task(task, miner_hotkey)
                all_period_scores = [ps for ps in period_scores if ps.period_score is not None]
                normalised_period_score = WeightSetter._normalise_period_scores(all_period_scores)

                overall_score_for_task = combined_quality_score * normalised_period_score

                hotkey_to_overall_scores[miner_hotkey] = overall_score_for_task * volume

                bt.logging.info(
                    f"Got overall hotkey score: {hotkey_to_overall_scores[miner_hotkey]},\n The quality score for this task is {combined_quality_score} \nand the normalised period score is {normalised_period_score}. Volume is: {volume}"
                )

            sum_of_scores = sum(hotkey_to_overall_scores.values())
            if sum_of_scores == 0:
                continue
            normalised_scores_for_task = {
                hotkey: importance * score / sum_of_scores for hotkey, score in hotkey_to_overall_scores.items()
            }
            for hotkey in normalised_scores_for_task:
                total_hotkey_scores[hotkey] = total_hotkey_scores.get(hotkey, 0) + normalised_scores_for_task[hotkey]

            bt.logging.info(f"Normalised hotkeys scores for task: {task}\n{normalised_scores_for_task}")

        bt.logging.info(f"Total hotkey scores: {total_hotkey_scores}")
        return total_hotkey_scores

    @staticmethod
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
            return total_score / total_weight
