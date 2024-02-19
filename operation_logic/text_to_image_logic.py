
import bittensor as bt

from core import constants as cst
from core import resource_management
from core import utils as core_utils
from models import base_models, utility_models
from operation_logic import utils as operation_utils


async def text_to_image_logic(
    body: base_models.TextToImageIncoming,
) -> base_models.TextToImageOutgoing:
    """Add gpu potential"""

    resource_manager = resource_management.SingletonResourceManager()

    output = base_models.TextToImageOutgoing(image_b64s=[])


    negative_prompt_prefix = cst.DEFAULT_NEGATIVE_PROMPT

    if body.engine == utility_models.EngineEnum.SDXL_TURBO:
        pipe, _ = resource_manager.get_resource(cst.MODEL_SDXL_TURBO)
    elif body.engine == utility_models.EngineEnum.KANDINSKY_22:
        pipe = resource_manager.get_resource(cst.MODEL_KANDINSKY)
        negative_prompt_prefix = cst.KANDINKSY_NEGATIVE_PROMPT_PERFIX + negative_prompt_prefix

    else:
        raise NotImplementedError(f"Engine {body.engine} not implemented")

    positive_prompt, negative_prompt = operation_utils.get_positive_and_negative_prompts(body.text_prompts)

    bt.logging.info(f"Using the seed: {body.seed}")
    seed_generator = operation_utils.get_seed_generator(body.seed)

    optional_kwargs = {}
    if body.height is not None and body.width is not None:
        optional_kwargs["height"] = body.height
        optional_kwargs["width"] = body.width

    processed_image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt_prefix + negative_prompt,
        guidance_scale=body.cfg_scale,
        num_inference_steps=body.steps,
        generator=seed_generator,
        **optional_kwargs,
    ).images[0]

    if operation_utils.image_is_nsfw(processed_image):
        bt.logging.info("NSFW image detected 👿, returning a corresponding error and no image")
        output.error_message = cst.NSFW_RESPONSE_ERROR
        return output

    bt.logging.info("✅ Took an image and made an image 😎")
    output.image_b64s = [core_utils.get_b64_from_pipeline_image(processed_image)]
    return output
