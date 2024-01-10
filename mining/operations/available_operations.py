from typing import Tuple

import bittensor as bt

from core import resource_management
from mining import core_miner
from mining.operations import abstract_operation
from models import protocols

operation_name = "AvailableOperations"


class AvailableOperations(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.AvailableOperations) -> protocols.AvailableOperations:
        # This needs to be defined in some config somewhere
        synapse.available_operations = resource_management.SingletonResourceManager().get_available_operations()
        bt.logging.info(f"Available operations: {synapse.available_operations}")
        return synapse

    @staticmethod
    def blacklist(synapse: protocols.AvailableOperations) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.AvailableOperations) -> float:
        return core_miner.base_priority(synapse)
