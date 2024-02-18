import pytest

import utils as testing_utils
from core import dataclasses as dc
from models import base_models, utility_models
from operation_logic import scribble_logic

sd_15_params = [
    base_models.ScribbleIncoming(
        engine=utility_models.EngineEnum.SDXL_1_5,
        text_prompts=[dc.TextPrompt(text="A green backpack", weight=1.0)],
        cfg_scale=5,
        steps=13,
        seed=42,
        height=1024,
        width=1024,
        image_strength=0.75,
        init_image=testing_utils.get_testing_image("test_scribble"),
    ),
    base_models.ScribbleIncoming(
        engine=utility_models.EngineEnum.SDXL_1_5,
        text_prompts=[dc.TextPrompt(text="A blue backpack"), dc.TextPrompt(text="handles", weight=-1.0)],
        cfg_scale=3,
        steps=25,
        seed=10,
        height=1024,
        width=1024,
        init_image=testing_utils.get_testing_image("test_scribble"),
    ),
]


@pytest.mark.parametrize("body", sd_15_params)
@pytest.mark.asyncio
async def test_image_to_image_logic_kandinsky(test_config_scribble, body):
    output = await scribble_logic.scribble_logic(body)
    assert len(output.image_b64s) > 0

    for b64_img in output.image_b64s:
        testing_utils.save_image(action="scribble", model_name="scribble", cfg_scale=body.cfg_scale, steps=body.steps, image_b64=b64_img)
