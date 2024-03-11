
import bittensor as bt

from core import constants as cst
from core import resource_management
from core import utils as core_utils
from models import base_models, utility_models
from operation_logic import utils as operation_utils
import httpx
import httpx
import json
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import routing
from typing import Optional
from operation_logic.sota import goapi, imagine_slash

async def sota_logic(
    body: base_models.SotaIncoming,
) -> base_models.SotaOutgoing:
    
    resource_manager = resource_management.SingletonResourceManager()

    output = base_models.SotaOutgoing()
    bt.logging.info("here 1")

    SOTA_provider = cst.PROVIDER_INT_TO_NAME.get(int(resource_manager.get_resource(cst.MODEL_SOTA)), None)

    bt.logging.info(f"SOTA provider: {SOTA_provider}")

    bt.logging.info("here 2")
    if SOTA_provider is None:
        bt.logging.error(f"You're serving the synapse but with no provider?! How as that happened")
    elif SOTA_provider == cst.GO_API_PROVIDER:
        bt.logging.info("here 3")
        image_url = await goapi.get_image(body.prompt)
        output.image_url = image_url
    elif  SOTA_provider == cst.IMAGINE_SLASH_PROVIDER:
        image_url = await imagine_slash.get_image(body.prompt)
        output.image_url = image_url

    
    return output
