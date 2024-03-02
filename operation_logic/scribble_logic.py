import base64
import binascii
import io

from PIL import Image

from core import constants as cst
from core import resource_management
from core import utils as core_utils
from models import base_models
from operation_logic import utils as operation_utils
import bittensor as bt

async def scribble_logic(body: base_models.ScribbleIncoming) -> base_models.ScribbleOutgoing:
    scribble_pipeline = resource_management.SingletonResourceManager().get_resource(cst.MODEL_SCRIBBLE)

    output = base_models.ScribbleOutgoing(image_b64s=[])

    try:
        img_bytes = base64.b64decode(body.init_image)
        image = Image.open(io.BytesIO(img_bytes))
    except binascii.Error:
        output.error_message = "Invalid b64 for the init image sent"
        return output

    positive_prompt, negative_prompt = operation_utils.get_positive_and_negative_prompts(body.text_prompts)

    seed_generator = operation_utils.get_seed_generator(body.seed)

    optional_kwargs = {}
    if body.height is not None and body.width is not None:
        optional_kwargs["height"] = operation_utils.get_closest_mutliple_of_64(body.height)
        optional_kwargs["width"] = operation_utils.get_closest_mutliple_of_64(body.width)
    else:
        image_width, image_height = image.size
        optional_kwargs["height"] = operation_utils.get_closest_mutliple_of_64(image_height)
        optional_kwargs["width"] = operation_utils.get_closest_mutliple_of_64(image_width)

    processed_image = scribble_pipeline(
        prompt=positive_prompt,
        guess_mode=positive_prompt == "",
        image=image,
        negative_prompt=cst.DEFAULT_NEGATIVE_PROMPT + negative_prompt,
        guidance_scale=body.cfg_scale,
        strength=1 - body.image_strength,
        num_inference_steps=body.steps,
        generator=seed_generator,
        **optional_kwargs
    ).images[0]

    image_hashes = operation_utils.image_hash_feature_extraction(processed_image)
    clip_embedding = operation_utils.get_clip_embedding_from_processed_image(processed_image)

    output.image_hashes = [image_hashes]
    output.clip_embeddings = [clip_embedding]


    if operation_utils.image_is_nsfw(processed_image):
        bt.logging.info("NSFW image detected 👿, returning a corresponding error and no image")
        output.error_message = cst.NSFW_RESPONSE_ERROR
        return output

    bt.logging.info("✅ Took a scribble and made a masterpiece 😎")
    output.image_b64s = [core_utils.get_b64_from_pipeline_image(processed_image)]
    return output
