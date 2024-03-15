from typing import Tuple, TypeVar

import bittensor as bt

from mining import core_miner
from mining.operations import abstract_operation
from models import base_models, protocols
from operation_logic import sota_logic

operation_name = "SotaOperation"

T = TypeVar("T", bound=bt.Synapse)


class SotaOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.Sota) -> protocols.Sota:
        output = await sota_logic.sota_logic(base_models.SotaIncoming(**synapse.dict()))
        output_dict = output.dict()
        for field in output_dict:
            setattr(synapse, field, output_dict[field])

        return synapse

    @staticmethod
    def blacklist(synapse: protocols.Sota) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.Sota) -> float:
        return core_miner.base_priority(synapse)
