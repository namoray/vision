
from models import base_models
import bittensor as bt
from operation_logic.sota import goapi
from config.miner_config import config as miner_config


async def sota_logic(
    body: base_models.SotaIncoming,
) -> base_models.SotaOutgoing:
    output = base_models.SotaOutgoing()

    Sota_key = miner_config.sota_provider_api_key
    if Sota_key is None:
        bt.logging.error("Sota key not set")
        output.error_message = "Sota key not set"
        return output

    image_url = await goapi.get_image(body.prompt, Sota_key)
    output.image_url = image_url

    if image_url is not None:
        bt.logging.info("✅ Successfully got a midjourney image!")
    return output
