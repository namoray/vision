from core import constants as cst
from core import resource_management
from models import base_models
import bittensor as bt
from operation_logic.sota import goapi

async def sota_logic(
    body: base_models.SotaIncoming,
) -> base_models.SotaOutgoing:
    
    resource_manager = resource_management.SingletonResourceManager()

    output = base_models.SotaOutgoing()

    Sota_key =resource_manager.get_resource(cst.MODEL_SOTA)
    
    image_url = await goapi.get_image(body.prompt, Sota_key)
    output.image_url = image_url

    if image_url is not None:
        bt.logging.info("✅ Successfully got a midjourney image!")
    return output
