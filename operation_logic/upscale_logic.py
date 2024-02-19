import base64
import binascii
import io
from PIL import Image

from core import constants as cst, utils as core_utils
from core import resource_management
from operation_logic import utils as operation_utils
from models import base_models
import bittensor as bt

MAX_PIX_COUNT = 4194304



async def upscale_logic(body: base_models.UpscaleIncoming) -> base_models.UpscaleOutgoing:
    """Add gpu potential"""

    upscale_model = resource_management.SingletonResourceManager().get_resource(cst.MODEL_UPSCALE)

    output = base_models.UpscaleOutgoing(image_b64s=[])


    try:
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes))
    except binascii.Error:
        output.error_message = "Invalid b64 for the image to upscale"
        return output


    upscaled_image = upscale_model.predict(image, batch_size=4)

    if operation_utils.image_is_nsfw(upscaled_image):
        bt.logging.info("NSFW image detected 👿, returning a corresponding error and no image")
        output.error_message = "Upscaled image is NSFW"
        return output

    bt.logging.info("✅ Took an image and made it crisper 😎")
    output.image_b64s = [core_utils.get_b64_from_pipeline_image(upscaled_image)]

    return output
