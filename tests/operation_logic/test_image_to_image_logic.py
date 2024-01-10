import pytest

import utils as testing_utils
from core import dataclasses as dc
from core import resource_management
from models import base_models, utility_models
from operation_logic import image_to_image_logic

kandinsky_params = [
    base_models.ImageToImageIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A Green flower in Rome", weight=1.0)],
        cfg_scale=3,
        steps=15,
        seed=42,
        height=1024,
        width=1024,
        init_image=testing_utils.get_testing_image("test_1024_1024_img2img"),
        image_strength=0.5,
    ),
    base_models.ImageToImageIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A demon in his lair"), dc.TextPrompt(text="Sunshine", weight=-1.0)],
        cfg_scale=3,
        steps=10,
        height=512,
        width=512,
        init_image=testing_utils.get_testing_image("test_512_512_img2img"),
        image_strength=0.7,
    ),
    base_models.ImageToImageIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A demon in his lair"), dc.TextPrompt(text="Sunshine", weight=-1.0)],
        cfg_scale=3,
        steps=25,
        height=712,
        width=1428,
        init_image=testing_utils.get_testing_image("test_512_512_img2img"),
        image_strength=0.7,
    ),
]

sdxl_turbo_params = [
    base_models.ImageToImageIncoming(
        engine=utility_models.EngineEnum.SDXL_TURBO,
        text_prompts=[dc.TextPrompt(text="A Green flower in Rome", weight=1.0)],
        cfg_scale=1.5,
        steps=20,
        seed=42,
        height=1024,
        width=1024,
        init_image=testing_utils.get_testing_image("test_1024_1024_img2img"),
        image_strength=0.2,
    ),
    base_models.ImageToImageIncoming(
        engine=utility_models.EngineEnum.SDXL_TURBO,
        text_prompts=[dc.TextPrompt(text="A demon in his lair"), dc.TextPrompt(text="Sunshine", weight=-1.0)],
        cfg_scale=1.0,
        steps=5,
        height=512,
        width=512,
        init_image=testing_utils.get_testing_image("test_512_512_img2img"),
        image_strength=0.7,
    ),
    base_models.ImageToImageIncoming(
        engine=utility_models.EngineEnum.SDXL_TURBO,
        text_prompts=[dc.TextPrompt(text="A demon in his lair"), dc.TextPrompt(text="Sunshine", weight=-1.0)],
        cfg_scale=1.0,
        steps=5,
        height=500,
        width=1500,
        init_image=testing_utils.get_testing_image("test_512_512_img2img"),
        image_strength=0.7,
    ),
]

@pytest.mark.parametrize("body", kandinsky_params)
@pytest.mark.asyncio
async def test_image_to_image_logic_kandinsky(test_config_kandinsky: resource_management.ResourceConfig, body):
    output = await image_to_image_logic.image_to_image_logic(body)
    assert len(output.image_b64s) > 0

    for b64_img in output.image_b64s:
        testing_utils.save_image(
            model_name="kandinsky",
            cfg_scale=body.cfg_scale,
            steps=body.steps,
            image_b64=b64_img,
            image_strength=body.image_strength,
        )


@pytest.mark.parametrize("body", sdxl_turbo_params)
@pytest.mark.asyncio
async def test_image_to_image_logic_sdxl_turbo(test_config_sdxl_turbo: resource_management.ResourceConfig, body):
    output = await image_to_image_logic.image_to_image_logic(body)
    assert len(output.image_b64s) > 0

    for b64_img in output.image_b64s:
        testing_utils.save_image(model_name='sdxl_turbo', cfg_scale=body.cfg_scale, steps=body.steps, image_b64=b64_img, image_strength=body.image_strength)
