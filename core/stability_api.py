from ast import Tuple
from PIL import Image
import aiohttp
import random
import base64
import os
import io
import bittensor as bt
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image
from io import BytesIO


from dotenv import load_dotenv
from core import constants as cst, dataclasses as dc

load_dotenv()

API_HOST = "https://api.stability.ai"
API_KEY = os.getenv("STABILITY_API_KEY")

if API_KEY is None:
    raise Exception("STABILITY_API_KEY is not set. Please run `export STABILITY_API_KEY=YOUR_API_KEY`")


def resize_image(image_b64: str) -> str:
    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data))
    
    best_size = find_closest_allowed_size(image)
    resized_image = image.resize(best_size, Image.Resampling.BICUBIC)
    
    byte_arr = BytesIO()
    resized_image.save(byte_arr, format='JPEG')
    encoded_resized_image = base64.b64encode(byte_arr.getvalue()).decode('utf-8')    
    return encoded_resized_image

def find_closest_allowed_size(image) -> Tuple[int, int]:
    width, height = image.size
    min_diff: float = float("inf")
    best_size: Tuple[int, int] = None
    for size in cst.ALLOWED_IMAGE_SIZES:
        diff = abs(width - size[0]) + abs(height - size[1])
        if diff < min_diff:
            min_diff = diff
            best_size = size
    return best_size


async def generate_images_from_text(
    text_prompts: List[dc.TextPrompt],
    cfg_scale: int = cst.DEFAULT_CFG_SCALE,
    height: int = cst.DEFAULT_HEIGHT,
    width: int = cst.DEFAULT_WIDTH,
    samples: int = cst.DEFAULT_SAMPLES,
    steps: int = cst.DEFAULT_STEPS,
    style_preset: Optional[str] = cst.DEFAULT_STYLE_PRESET,
    seed: int = random.randint(1, cst.LARGEST_SEED),
    engine_id: str = cst.DEFAULT_ENGINE,
) -> List[str]:
    
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            f"{API_HOST}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "text_prompts": [text_prompt.dict() for text_prompt in text_prompts],
                "cfg_scale": cfg_scale,
                "height": height,
                "width": width,
                "samples": samples,
                "steps": steps,
                "seed": seed,
                "style_preset": style_preset,
            },
        )

        image_b64s = []
        if response.status != 200:
            response_json = await response.json()
            bt.logging.warning(f"USER ERROR: Bad response, with code {response.status} :( response json: {response_json}")
            return response_json

        response_json = await response.json()
        for i, image in enumerate(response_json.get("artifacts", [])):
            image_b64s.append(image["base64"])

        return image_b64s


async def generate_images_from_image(
    init_image: str,
    text_prompts: List[dc.TextPrompt],
    cfg_scale: int = cst.DEFAULT_CFG_SCALE,
    samples: int = cst.DEFAULT_SAMPLES,
    steps: int = cst.DEFAULT_STEPS,
    init_image_mode: str = cst.DEFAULT_INIT_IMAGE_MODE,
    image_strength: float = cst.DEFAULT_IMAGE_STRENGTH,
    style_preset: Optional[str] = cst.DEFAULT_STYLE_PRESET,
    sampler: Optional[str] = cst.DEFAULT_SAMPLER,
    seed: int = random.randint(1, cst.LARGEST_SEED),
    engine_id: str = cst.DEFAULT_ENGINE,
) -> List[str]:
    image_resized = resize_image(init_image)
    bt.logging.debug("Guess who just resized an image!")
    data = {
        "init_image": base64.b64decode(image_resized),
        "image_strength": str(image_strength),
        "init_image_mode": init_image_mode,
        "cfg_scale": str(cfg_scale),
        "samples": str(samples),
        "steps": str(steps),
        "seed": str(seed),
    }
    for i, prompt in enumerate(text_prompts):
        prompt_dict = prompt.dict()
        data[f"text_prompts[{i}][text]"] = prompt_dict["text"]
        data[f"text_prompts[{i}][weight]"] = str(prompt_dict["weight"])

    if style_preset:
        data["style_preset"] = style_preset
    
    if sampler:
        data["sampler"] = sampler

    bt.logging.debug("Sending request!")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_HOST}/v1/generation/{engine_id}/image-to-image",
            headers={"Accept": "application/json", "Authorization": f"Bearer {API_KEY}"},
            data=data,
        ) as response:
            image_b64s = []
            if response.status != 200:
                response_json = await response.json()
                bt.logging.warning(f"USER ERROR: Bad response, with code {response.status} :( response json: {response_json}")
                return image_b64s

            response_json = await response.json()
            for i, image in enumerate(response_json.get("artifacts", [])):
                image_b64s.append(image["base64"])

        return image_b64s


async def upscale_image(
    init_image: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> List[str]:

    data = {
        "image": base64.b64decode(init_image),
        "height": str(height),
        "width": str(width),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_HOST}/v1/generation/{cst.UPSCALE_ENGINE}/image-to-image/upscale",
            headers={"Accept": "application/json", "Authorization": f"Bearer {API_KEY}"},
            data=data,
        ) as response:
            image_b64s = []
            if response.status != 200:
                response_json = await response.json()
                bt.logging.warning(f"USER ERROR: Bad response, with code {response.status} :( response json: {response_json}")
                return image_b64s

            response_json = await response.json()
            for i, image in enumerate(response_json.get("artifacts", [])):
                image_b64s.append(image["base64"])

        return image_b64s