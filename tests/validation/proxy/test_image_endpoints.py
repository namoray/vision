import requests  # noqa
from models import request_models, utility_models  # noqa
from core import dataclasses as dc  # noqa
import base64  # noqa
from core import utils  # noqa


def image_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


init_image = image_to_base64("tests/assets/tall_man.webp")

BASE_URL = "http://localhost:8090/"
# Tests are pretty manual, because bittensor :D (need miners)


def test_text_to_image():
    data = request_models.TextToImageRequest(
        engine=utility_models.EngineEnum.PROTEUS.value,
        cfg_scale=2.0,
        steps=10,
        text_prompts=[
            dc.TextPrompt(
                text="Dog",
                weight=1.0,
            )
        ],
        height=1280,
        width=1280,
    )
    response = requests.post(BASE_URL + "text-to-image", json=data.dict())
    assert response.status_code == 200

    data = request_models.TextToImageRequest(
        engine=utility_models.EngineEnum.DREAMSHAPER.value,
        cfg_scale=2.0,
        steps=10,
        text_prompts=[
            dc.TextPrompt(
                text="Dog",
                weight=1.0,
            )
        ],
        height=1024,
        width=1024,
    )
    response = requests.post(BASE_URL + "text-to-image", json=data.dict())
    assert response.status_code == 200

    data = request_models.TextToImageRequest(
        engine=utility_models.EngineEnum.PLAYGROUND.value,
        cfg_scale=2.0,
        steps=25,
        text_prompts=[
            dc.TextPrompt(
                text="Dog",
                weight=1.0,
            )
        ],
        height=1024,
        width=1024,
    )
    response = requests.post(BASE_URL + "text-to-image", json=data.dict())
    assert response.status_code == 200


def test_image_to_image():
    data = request_models.ImageToImageRequest(
        engine=utility_models.EngineEnum.PROTEUS.value,
        cfg_scale=2.0,
        steps=8,
        text_prompts=[
            dc.TextPrompt(
                text="Cow",
                # weight=1.0,
            )
        ],
        image_strength=0.5,
        init_image=init_image,
        height=1280,
        width=1280,
    )
    response = requests.post(BASE_URL + "image-to-image", json=data.dict())
    assert response.status_code == 200


data = request_models.ImageToImageRequest(
    engine=utility_models.EngineEnum.DREAMSHAPER.value,
    cfg_scale=2.0,
    steps=10,
    text_prompts=[
        dc.TextPrompt(
            text="Dog",
            weight=1.0,
        )
    ],
    image_strength=0.5,
    init_image=init_image,
    height=1280,
    width=1280,
)
response = requests.post(BASE_URL + "image-to-image", json=data.dict())
assert response.status_code == 200

data = request_models.ImageToImageRequest(
    engine=utility_models.EngineEnum.PLAYGROUND.value,
    cfg_scale=2.0,
    steps=25,
    text_prompts=[
        dc.TextPrompt(
            text="Dog",
            weight=1.0,
        )
    ],
    image_strength=0.5,
    init_image=init_image,
    height=1280,
    width=1280,
)
response = requests.post(BASE_URL + "image-to-image", json=data.dict())
assert response.status_code == 200


def test_inpaint():
    data = request_models.InpaintRequest(
        cfg_scale=2.0,
        steps=8,
        text_prompts=[
            dc.TextPrompt(
                text="Dog",
                weight=1.0,
            )
        ],
        init_image=init_image,
        mask_image=utils.generate_mask_with_circle(init_image),
    )
    response = requests.post(BASE_URL + "inpaint", json=data.dict())
    print(response.text)
    assert response.status_code == 200


# def test_avatar():
#     data = request_models.AvatarRequest(
#         steps=8,
#         text_prompts=[
#             dc.TextPrompt(
#                 text="Dog",
#                 weight=1.0,
#             )
#         ],
#         init_image=init_image,
#         ipadapter_strength=0.5,
#         control_strength=0.5,
#         height=1024,
#         width=1024,
#     )
#     response = requests.post(BASE_URL + "avatar", json=data.dict())
#     assert response.status_code == 200, response.content
