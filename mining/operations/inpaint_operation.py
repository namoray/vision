from typing import Tuple, TypeVar

import bittensor as bt

from mining import core_miner
from mining.operations import abstract_operation
from models import base_models, protocols
from operation_logic import inpaint_logic

operation_name = "InpaintOperation"

T = TypeVar("T", bound=bt.Synapse)


class InpaintOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.Inpaint) -> protocols.Inpaint:
        output = await inpaint_logic.inpaint_logic(base_models.InpaintIncoming(**synapse.dict()))

        synapse.init_image = None
        synapse.mask_image = None

        synapse.image_b64s = output.image_b64s
        synapse.error_message = output.error_message

        return synapse

    @staticmethod
    def blacklist(synapse: protocols.Inpaint) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.Inpaint) -> float:
        return core_miner.base_priority(synapse)
