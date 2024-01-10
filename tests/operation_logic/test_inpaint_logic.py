import pytest

import utils as testing_utils
from core import dataclasses as dc
from models import base_models, utility_models
from operation_logic import inpaint_logic

kandinsky_params = [
    base_models.InpaintIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A Green flower in Rome", weight=1.0)],
        cfg_scale=3,
        steps=40,
        seed=42,
        init_image=testing_utils.get_testing_image("test_inpaint_before"),
        mask_image=testing_utils.get_testing_image("test_inpaint_mask"),
        height=1024,
        width=1024,
    ),
    base_models.InpaintIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A demon in his lair"), dc.TextPrompt(text="Sunshine", weight=-1.0)],
        cfg_scale=3,
        steps=40,
        seed=1234,
        init_image=testing_utils.get_testing_image("test_inpaint_before"),
        mask_image=testing_utils.get_testing_image("test_inpaint_mask"),
        height=512,
        width=512,
    ),
    base_models.InpaintIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A sunflower in  a meadow")],
        cfg_scale=3,
        steps=40,
        seed=1234,
        init_image=testing_utils.get_testing_image("test_inpaint_feather_image"),
        mask_image=testing_utils.get_testing_image("test_inpaint_feather_mask"),
    ),
    base_models.InpaintIncoming(
        engine=utility_models.EngineEnum.KANDINSKY_22,
        text_prompts=[dc.TextPrompt(text="A sunflower in  a meadow")],
        cfg_scale=3,
        steps=40,
        seed=1234,
        init_image=testing_utils.get_testing_image("test_inpaint_feather_image"),
        mask_image=testing_utils.get_testing_image("test_inpaint_feather_mask"),
        height=1024,
        width=1024,
    ),
]



@pytest.mark.parametrize("body", kandinsky_params)
@pytest.mark.asyncio
async def test_inpaint_logic_kandinsky(test_config_kandinsky, body):
    output = await inpaint_logic.inpaint_logic(body)
    assert len(output.image_b64s) > 0

    for b64_img in output.image_b64s:
        testing_utils.save_image(action="inpaint", model_name="kandinsky", cfg_scale=body.cfg_scale, steps=body.steps, image_b64=b64_img)
