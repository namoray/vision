from typing import Tuple, TypeVar

import bittensor as bt

from mining import core_miner
from mining.operations import abstract_operation
from models import base_models, protocols
from operation_logic import upscale_logic

operation_name = "UpscaleOperation"

T = TypeVar("T", bound=bt.Synapse)


class UpscaleOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.Upscale) -> protocols.Upscale:
        output = await upscale_logic.upscale_logic(base_models.UpscaleIncoming(**synapse.dict()))

        synapse.image = None

        synapse.image_b64s = output.image_b64s
        synapse.error_message = output.error_message
        return synapse

    @staticmethod
    def blacklist(synapse: protocols.Upscale) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.Upscale) -> float:
        return core_miner.base_priority(synapse)
