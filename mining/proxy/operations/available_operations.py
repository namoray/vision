from typing import Tuple

import bittensor as bt

from mining.proxy import core_miner
from mining.proxy.operations import abstract_operation
from models import synapses
from core import tasks
operation_name = "AvailableTasksOperation"


class AvailableTasksOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(
        synapse: synapses.AvailableTasksOperation,
    ) -> synapses.AvailableTasksOperation:

        synapse.available_tasks = tasks.SUPPORTED_TASKS
        bt.logging.info(f"✅ Available operations: {synapse.available_tasks}")
        return synapse

    @staticmethod
    def blacklist(synapse: synapses.AvailableTasksOperation) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: synapses.AvailableTasksOperation) -> float:
        return core_miner.base_priority(synapse)
