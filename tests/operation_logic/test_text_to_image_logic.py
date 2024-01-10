import pytest

import utils as testing_utils
from core import dataclasses as dc
from models import base_models, utility_models
from operation_logic import text_to_image_logic

kandinsky_params = [
    base_models.TextToImageIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A cat with a frown :(", weight=1.0)],
        cfg_scale=1.5,
        height=512,
        width=512,
        steps=40,
        seed=42,
    ),
    base_models.TextToImageIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A cat with no eyes"), dc.TextPrompt(text="Dog", weight=-1.0)],
        cfg_scale=1.0,
        height=1024,
        width=1024,
        steps=10,
    ),
]

sdxl_turbo_params = [
    base_models.TextToImageIncoming(
        engine=utility_models.EngineEnum.SDXL_TURBO,
        text_prompts=[dc.TextPrompt(text="A cat with a frown :(", weight=1.0)],
        cfg_scale=1.5,
        height=512,
        width=512,
        steps=40,
        seed=42,
    ),
    base_models.TextToImageIncoming(
        engine=utility_models.EngineEnum.SDXL_TURBO,
        text_prompts=[dc.TextPrompt(text="A cat with no eyes"), dc.TextPrompt(text="Dog", weight=-1.0)],
        cfg_scale=1.0,
        height=1024,
        width=1024,
        steps=10,
    ),
]

@pytest.mark.parametrize("body", kandinsky_params)
@pytest.mark.asyncio
async def test_text_to_image_logic_kandinsky(test_config_kandinsky, body):
    output = await text_to_image_logic.text_to_image_logic(body)
    assert len(output.image_b64s) > 0

    for b64_img in output.image_b64s:
        testing_utils.save_image(model_name="kandinsky", cfg_scale=body.cfg_scale, steps=body.steps, image_b64=b64_img)


@pytest.mark.parametrize("body", sdxl_turbo_params)
@pytest.mark.asyncio
async def test_text_to_image_logic_sdxl_turbo(test_config_sdxl_turbo, body):
    output = await text_to_image_logic.text_to_image_logic(body)
    assert len(output.image_b64s) > 0

    for b64_img in output.image_b64s:
        testing_utils.save_image(model_name='sdxl_turbo', cfg_scale=body.cfg_scale, steps=body.steps, image_b64=b64_img)
