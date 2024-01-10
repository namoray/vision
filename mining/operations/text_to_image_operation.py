from typing import Tuple, TypeVar

import bittensor as bt

from mining import core_miner
from mining.operations import abstract_operation
from models import base_models, protocols
from operation_logic import text_to_image_logic

operation_name = "TextToImageOperation"

T = TypeVar("T", bound=bt.Synapse)


class TextToImageOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.TextToImage) -> protocols.TextToImage:
        output = await text_to_image_logic.text_to_image_logic(base_models.TextToImageIncoming(**synapse.dict()))

        synapse.image_b64s = output.image_b64s
        synapse.error_message = output.error_message

        return synapse

    @staticmethod
    def blacklist(synapse: protocols.TextToImage) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.TextToImage) -> float:
        return core_miner.base_priority(synapse)
