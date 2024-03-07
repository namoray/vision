from typing import Tuple, TypeVar

import bittensor as bt

from mining import core_miner
from mining.operations import abstract_operation
from models import base_models, protocols
from operation_logic import clip_embeddings_logic

# Make sure this matches the class name
operation_name = "ClipEmbeddingsOperation"

T = TypeVar("T", bound=bt.Synapse)


class ClipEmbeddingsOperation(abstract_operation.Operation):
    @staticmethod
    async def forward(synapse: protocols.ClipEmbeddings) -> protocols.ClipEmbeddings:
        output = await clip_embeddings_logic.clip_embeddings_logic(base_models.ClipEmbeddingsIncoming(**synapse.dict()))

        synapse.image_b64s = None

        output_dict = output.dict()
        for field in output_dict:
            setattr(synapse, field, output_dict[field])

        return synapse

    @staticmethod
    def blacklist(synapse: protocols.ClipEmbeddings) -> Tuple[bool, str]:
        return core_miner.base_blacklist(synapse)

    @staticmethod
    def priority(synapse: protocols.ClipEmbeddings) -> float:
        return core_miner.base_priority(synapse)
