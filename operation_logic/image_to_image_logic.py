import base64
import binascii
import io

from PIL import Image

from core import constants as cst
from core import resource_management
from core import utils as core_utils
from models import base_models, utility_models
from operation_logic import utils as operation_utils
import bittensor as bt

async def image_to_image_logic(
    body: base_models.ImageToImageIncoming,
) -> base_models.ImageToImageOutgoing:
    """Add gpu potential"""

    output = base_models.ImageToImageOutgoing(image_b64s=[])

    negative_prompt_prefix = cst.DEFAULT_NEGATIVE_PROMPT

    if body.engine == utility_models.EngineEnum.SDXL_TURBO:
        _, pipe = resource_management.SingletonResourceManager().get_resource(cst.MODEL_SDXL_TURBO)
    elif body.engine == utility_models.EngineEnum.KANDINSKY_22:
        pipe = resource_management.SingletonResourceManager().get_resource(cst.MODEL_KANDINSKY)
        negative_prompt_prefix = cst.KANDINKSY_NEGATIVE_PROMPT_PERFIX + negative_prompt_prefix
    else:
        raise NotImplementedError(f"Engine {body.engine} not implemented")


    positive_prompt, negative_prompt = operation_utils.get_positive_and_negative_prompts(body.text_prompts)

    try:
        img_bytes = base64.b64decode(body.init_image)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except binascii.Error:
        output.error_message = "Invalid b64 init image sent. Please ensure you are sending a valid base64 encoded image."

    seed_generator = operation_utils.get_seed_generator(body.seed)

    if body.height is not None and body.width is not None:
        height = operation_utils.get_closest_mutliple_of_64(body.height)
        width = operation_utils.get_closest_mutliple_of_64(body.width)
    else:
        image_width, image_height = image.size
        height = operation_utils.get_closest_mutliple_of_64(image_height)
        width = operation_utils.get_closest_mutliple_of_64(image_width)

    image_strength = max(1 - body.image_strength, 0.01)

    processed_image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt_prefix + negative_prompt,
        guidance_scale=body.cfg_scale,
        image=image,
        strength=image_strength,
        num_inference_steps=body.steps,
        generator=seed_generator,
        height=height,
        width=width,
    ).images[0]

    image_hashes = operation_utils.image_hash_feature_extraction(processed_image)
    clip_embedding = operation_utils.get_clip_embedding_from_processed_image(processed_image)

    output.image_hashes = [image_hashes]
    output.clip_embeddings = [clip_embedding]


    if operation_utils.image_is_nsfw(processed_image):
        bt.logging.info("NSFW image detected 👿, returning a corresponding error and no image")
        output.error_message = cst.NSFW_RESPONSE_ERROR
        return output

    bt.logging.info("✅ Took an image and made an image 😎")

    output.image_b64s = [core_utils.get_b64_from_pipeline_image(processed_image)]
    return output
