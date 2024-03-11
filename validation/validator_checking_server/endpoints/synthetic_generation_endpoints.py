import random
from fastapi import APIRouter
from typing import List
from core import constants
from core import constants as core_cst
from core import dataclasses as dc
from core import resource_management
from models import base_models, utility_models, request_models
from operation_logic import utils as operation_utils
from validation.validator_checking_server import utils

router = APIRouter()


def generate_params(engine: utility_models.EngineEnum, params_to_vary: List[str]):
    params = {}

    for param in request_models.ALLOWED_PARAMS_FOR_ENGINE[engine]:
        if param in params_to_vary:
            value = request_models.ALLOWED_PARAMS_FOR_ENGINE[engine][param]["generator"]()
            params[param] = value
    return params


@router.get(f"/{core_cst.SYNTHETIC_ENDPOINT_PREFIX}/text-to-image")
async def text_to_image() -> base_models.TextToImageIncoming:

    positive_text = utils.get_markov_short_sentence()
    text_prompts = [dc.TextPrompt(text=positive_text, weight=1.0)]
    seed = random.randint(1, constants.LARGEST_SEED)

    engine = random.choice(
        [utility_models.EngineEnum.KANDINSKY_22.value, utility_models.EngineEnum.SDXL_TURBO.value ]
    )

    PARAMS_TO_VARY = [
        "steps", "height", "width", "cfg_scale"
    ]
    params = generate_params(engine, PARAMS_TO_VARY)


    return base_models.TextToImageIncoming(text_prompts=text_prompts, seed=seed, engine=engine, **params)


@router.get(f"/{core_cst.SYNTHETIC_ENDPOINT_PREFIX}/image-to-image")
async def image_to_image() -> base_models.ImageToImageIncoming:


    cache = resource_management.SingletonResourceManager().get_resource(core_cst.MODEL_CACHE)

    positive_text = utils.get_markov_short_sentence()
    text_prompts = [dc.TextPrompt(text=positive_text, weight=1.0)]
    seed = random.randint(1, constants.LARGEST_SEED)

    engine = random.choice(
        [utility_models.EngineEnum.KANDINSKY_22.value, utility_models.EngineEnum.SDXL_TURBO.value]
    )

    init_image = await utils.get_random_image_b64(cache)

    PARAMS_TO_VARY = [
        "steps", "height", "width", "cfg_scale", "image_strength"
    ]
    params = generate_params(engine, PARAMS_TO_VARY)

    return base_models.ImageToImageIncoming(text_prompts=text_prompts, init_image=init_image, seed=seed, engine=engine, **params)


@router.get(f"/{core_cst.SYNTHETIC_ENDPOINT_PREFIX}/inpaint")
async def inpaint() -> base_models.InpaintIncoming:

    cache = resource_management.SingletonResourceManager().get_resource(core_cst.MODEL_CACHE)

    positive_text = utils.get_markov_short_sentence()
    text_prompts = [dc.TextPrompt(text=positive_text, weight=1.0)]
    seed = random.randint(1, constants.LARGEST_SEED)

    init_image = await utils.get_random_image_b64(cache)
    mask_image = utils.generate_mask_with_circle(init_image)

    engine = utility_models.EngineEnum.KANDINSKY_22.value


    PARAMS_TO_VARY = [
        "steps", "cfg_scale",
    ]
    params = generate_params(engine, PARAMS_TO_VARY)

    params["steps"] = min(params["steps"], 25)


    return base_models.InpaintIncoming(
        text_prompts=text_prompts,
        init_image=init_image,
        mask_image=mask_image,
        seed=seed,
        engine=engine,
        **params,
    )


@router.get(f"/{core_cst.SYNTHETIC_ENDPOINT_PREFIX}/scribble")
async def scribble() -> base_models.ScribbleIncoming:

    cache = resource_management.SingletonResourceManager().get_resource(core_cst.MODEL_CACHE)

    raw_image_b64 = await utils.get_random_image_b64(cache)
    init_image_b64 = utils.generate_mask_with_circle(raw_image_b64)

    positive_text = utils.get_markov_short_sentence()
    text_prompts = [dc.TextPrompt(text=positive_text, weight=1.0)]
    seed = random.randint(1, constants.LARGEST_SEED)
    engine = utility_models.EngineEnum.SDXL_1_5.value

    PARAMS_TO_VARY = [
        "steps", "cfg_scale", "height", "width", "image_strength"
    ]
    params = generate_params(engine, PARAMS_TO_VARY)


    return base_models.ScribbleIncoming(init_image=init_image_b64, text_prompts=text_prompts, seed=seed, **params)


@router.get(f"/{core_cst.SYNTHETIC_ENDPOINT_PREFIX}/upscale")
async def upscale() -> base_models.UpscaleIncoming:

    cache = resource_management.SingletonResourceManager().get_resource(core_cst.MODEL_CACHE)
    image = await utils.get_random_image_b64(cache)

    image = operation_utils.resize_image(image)

    return base_models.UpscaleIncoming(image=image)


@router.get(f"/{core_cst.SYNTHETIC_ENDPOINT_PREFIX}/clip-embeddings")
async def clip_embeddings() -> base_models.ClipEmbeddingsIncoming:

    cache = resource_management.SingletonResourceManager().get_resource(core_cst.MODEL_CACHE)
    random_image_b64 = await utils.get_random_image_b64(cache)
    return base_models.ClipEmbeddingsIncoming(image_b64s=[random_image_b64])


@router.get(f"/{core_cst.SYNTHETIC_ENDPOINT_PREFIX}/sota")
async def text_to_image() -> base_models.SotaIncoming:

    positive_text = utils.get_markov_short_sentence()
    seed = random.randint(1, constants.LARGEST_SEED)


    positive_text += f"--seed {seed} --ar 1:1"


    return base_models.SotaIncoming(prompt=positive_text)