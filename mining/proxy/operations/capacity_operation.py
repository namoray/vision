from typing import Tuple, TypeVar

import bittensor as bt

from mining.proxy import core_miner
from mining.proxy.operations import abstract_operation
from models import synapses
from config.miner_config import config as miner_config
import copy

operation_name = "CapacityOperation"

T = TypeVar("T", bound=bt.Synapse)


class CapacityOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: synapses.Capacity) -> synapses.Capacity:
        capacities_with_concurrencies = copy.deepcopy(miner_config.capacity_config)
        for key in capacities_with_concurrencies:
            del capacities_with_concurrencies[key]["concurrency_group_id"]

        return synapses.Capacity(capacities=capacities_with_concurrencies)

    @staticmethod
    async def blacklist(synapse: synapses.Capacity) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    async def priority(synapse: synapses.Capacity) -> float:
        return core_miner.base_priority(synapse)
