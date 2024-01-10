import pytest

import utils as testing_utils
from models import base_models
from operation_logic import upscale_logic

realesrgan_params = [
    base_models.UpscaleIncoming(
        image=testing_utils.get_testing_image("test_upscale_before"),
    ),
    base_models.UpscaleIncoming(
        image=testing_utils.get_testing_image("test_upscale_before"),
    ),
    base_models.UpscaleIncoming(
        image=testing_utils.get_testing_image("test_upscale_before"),
    ),
    base_models.UpscaleIncoming(
        image=testing_utils.get_testing_image("test_upscale_before"),
    ),
]


@pytest.mark.parametrize("body", realesrgan_params)
@pytest.mark.asyncio
async def test_upscale_logic(test_config_realesrgan, body):
    output = await upscale_logic.upscale_logic(body)
    assert len(output.image_b64s) > 0
