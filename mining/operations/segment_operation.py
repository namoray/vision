from typing import Tuple, TypeVar

import bittensor as bt

from mining import core_miner
from mining.operations import abstract_operation
from models import base_models, protocols
from operation_logic import segment_logic

# Make sure this matches the class name
operation_name = "SegmentOperation"

T = TypeVar("T", bound=bt.Synapse)


class SegmentOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.Segment) -> protocols.Segment:
        output = await segment_logic.segment_logic(base_models.SegmentIncoming(**synapse.dict()))

        synapse.image_b64 = None

        # synapse.image_uuid = output.image_uuid if output.image_uuid is not None else synapse.image_uuid
        synapse.masks = output.masks
        synapse.image_shape = output.image_shape
        synapse.error_message = output.error_message

        return synapse

    @staticmethod
    def blacklist(synapse: protocols.Segment) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.Segment) -> float:
        return core_miner.base_priority(synapse)
