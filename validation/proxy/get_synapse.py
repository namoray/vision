from typing import Type
from pydantic import BaseModel
import bittensor as bt
from core import utils as core_utils, constants as core_cst
from validation.core_validator import core_validator


def get_synapse_from_body(
    body: BaseModel,
    synapse_model: Type[bt.Synapse],
) -> bt.Synapse:
    body_dict = body.dict()
    # I hate using the global var of core_validator as much as you hate reading it... gone in rewrite
    body_dict["seed"] = core_utils.get_seed(core_cst.SEED_CHUNK_SIZE, core_validator.validator_uid)
    synapse = synapse_model(**body_dict)
    return synapse
