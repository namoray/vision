from typing import Tuple, TypeVar

import bittensor as bt

from mining import core_miner
from mining.operations import abstract_operation
from models import base_models, protocols
from operation_logic import image_to_image_logic

operation_name = "ImageToImageOperation"

T = TypeVar("T", bound=bt.Synapse)


class ImageToImageOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.ImageToImage) -> protocols.ImageToImage:
        output = await image_to_image_logic.image_to_image_logic(base_models.ImageToImageIncoming(**synapse.dict()))

        synapse.init_image = None

        synapse.image_b64s = output.image_b64s
        synapse.error_message = output.error_message

        return synapse

    @staticmethod
    def blacklist(synapse: protocols.ImageToImage) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.ImageToImage) -> float:
        return core_miner.base_priority(synapse)
