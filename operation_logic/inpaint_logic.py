import base64
import binascii
import io

import bittensor as bt
from PIL import Image

from core import constants as cst
from core import resource_management
from core import utils as core_utils
from models import base_models, utility_models
from operation_logic import utils as operation_utils


async def inpaint_logic(body: base_models.InpaintIncoming) -> base_models.InpaintOutgoing:
    """Add gpu potential"""

    bt.logging.debug(f"Steps: {body.steps}")

    output = base_models.ImageToImageOutgoing(image_b64s=[])

    negative_prompt_prefix = cst.DEFAULT_NEGATIVE_PROMPT

    if body.engine == utility_models.EngineEnum.KANDINSKY_22:
        pipeline = resource_management.SingletonResourceManager().get_resource(cst.MODEL_KANDINSKY)
        negative_prompt_prefix = cst.KANDINKSY_NEGATIVE_PROMPT_PERFIX + negative_prompt_prefix
    else:
        raise NotImplementedError(f"Engine {body.engine} not implemented")

    output = base_models.InpaintOutgoing()



    try:
        img_bytes = base64.b64decode(body.init_image)
        image = Image.open(io.BytesIO(img_bytes))
    except binascii.Error:
        output.error_message = "Bad image b64 for inpaint. Please ensure that your image b64 is correct."
        return output
    try:
        mask_img_bytes = base64.b64decode(body.mask_image)
        mask_image = Image.open(io.BytesIO(mask_img_bytes))
    except binascii.Error:
        output.error_message = "Bad mask image b64 for inpaint. Please ensure that your image b64 is correct."
        return output

    image = operation_utils.pad_image_pil(image, 64, (255, 255, 255))

    if mask_image.mode == "L":
        border_color = 255
    else:
        border_color = (255, 255, 255)

    mask_image = operation_utils.pad_image_pil(mask_image, 64, border_color)

    image_width, image_height = image.size
    mask_image_width, mask_image_height = mask_image.size

    if image_width != mask_image_width or image_height != mask_image_height:
        output.error_message = "Mask image must be the same size as the image."
        return output

    positive_prompt, negative_prompt = operation_utils.get_positive_and_negative_prompts(body.text_prompts)

    seed_generator = operation_utils.get_seed_generator(body.seed)

    processed_image = pipeline(
        prompt=positive_prompt,
        image=image,
        mask_image=mask_image,
        negative_prompt=negative_prompt_prefix + negative_prompt,
        guidance_scale=body.cfg_scale,
        num_inference_steps=body.steps,
        generator=seed_generator,
        height=image_height,
        width=image_width,
    ).images[0]

    image_hashes = operation_utils.image_hash_feature_extraction(processed_image)
    clip_embedding = operation_utils.get_clip_embedding_from_processed_image(processed_image)

    output.image_hashes = [image_hashes]
    output.clip_embeddings = clip_embedding

    if operation_utils.image_is_nsfw(processed_image):
        bt.logging.info("NSFW image detected 👿, returning a corresponding error and no image")
        output.error_message = cst.NSFW_RESPONSE_ERROR
        return output

    bt.logging.info("✅ Took an image & did some painting 😎")

    output.image_b64s = [core_utils.get_b64_from_pipeline_image(processed_image)]
    return output
