# Schema for the db
import asyncio
import time
from typing import Dict, List, Tuple, Union

import bittensor as bt
import torch
from core import Task
from models import utility_models
from validation.models import axon_uid
from validation.weight_setting import calculations

VERSION_KEY = 40_002

# If 1 then only take into account the most recent. If zero then they are all equal
PERIOD_SCORE_TIME_DECAYING_FACTOR = 0.2


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
        task_weights: Dict[Task, float],
    ) -> None:
        total_hotkey_scores = calculations.calculate_scores_for_settings_weights(
            capacities_for_tasks, uid_to_uid_info, task_weights
        )

        processed_weight_uids, processed_weights = self._get_processed_weights_and_uids(
            metagraph, netuid, total_hotkey_scores, uid_to_uid_info
        )
        if processed_weight_uids is None:
            bt.logging.warning("Scores all zero, not setting weights!")
            return
        await asyncio.to_thread(self._set_weights, wallet, netuid, processed_weight_uids, processed_weights)

    def _get_processed_weights_and_uids(
        self,
        metagraph: bt.metagraph,
        netuid: int,
        total_hotkey_scores: Dict[str, float],
        uid_to_uid_info: Dict[axon_uid, utility_models.UIDinfo],
    ) -> Union[Tuple[Dict[str, float], List[axon_uid]], Tuple[None, None]]:
        hotkey_to_uid = {uid_info.hotkey: uid_info.uid for uid_info in uid_to_uid_info.values()}
        weights_tensor = torch.zeros_like(metagraph.S, dtype=torch.float32)
        for hotkey, score in total_hotkey_scores.items():
            uid = hotkey_to_uid[hotkey]
            weights_tensor[uid] = score

        if all(score == 0 for score in total_hotkey_scores.values()):
            return None, None
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

        return processed_weights, processed_weight_uids

    def _set_weights(
        self,
        wallet: bt.wallet,
        netuid: int,
        processed_weights: bt.Tensor,
        processed_weight_uids: bt.Tensor,
    ) -> None:
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
