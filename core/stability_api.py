from PIL import Image
import aiohttp
import random
import base64
import os
import io
import bittensor as bt
from typing import List, Dict, Optional, Union
from core import constants as cst, dataclasses as dc

API_HOST = "https://api.stability.ai"
API_KEY = os.getenv("STABILITY_API_KEY")




if API_KEY is None:
    raise Exception("STABILITY_API_KEY is not set. Please run `export STABILITY_API_KEY=YOUR_API_KEY`")


async def generate_images_from_text(
    text_prompts: List[dc.TextPrompt],
    engine_id: str = "stable-diffusion-v1-6",
    cfg_scale: int = cst.DEFAULT_CFG_SCALE,
    height: int = cst.DEFAULT_HEIGHT,
    width: int = cst.DEFAULT_WIDTH,
    samples: int = cst.DEFAULT_SAMPLES,
    steps: int = cst.DEFAULT_STEPS,
    style_preset: str = cst.DEFAULT_STYLE_PRESET,
    seed: int = random.randint(1, cst.LARGEST_SEED)
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
            bt.logging.warning("Bad response code :( {} {}".format(response.status, response.reason))

        response_json = await response.json()
        for i, image in enumerate(response_json.get("artifacts", [])):
            image_b64s.append(image["base64"])

        return image_b64s

async def generate_images_from_image(
    init_image: str,
    text_prompts: List[Dict[str, Union[str, float]]],
    cfg_scale: int = cst.DEFAULT_CFG_SCALE,
    samples: int = cst.DEFAULT_SAMPLES,
    steps: int = cst.DEFAULT_STEPS,
    init_image_mode: str = cst.DEFAULT_INIT_IMAGE_MODE,
    image_strength: float = cst.DEFAULT_IMAGE_STRENGTH,
    style_preset: Optional[str] = cst.DEFAULT_STYLE_PRESET,
    seed: int = random.randint(1, cst.LARGEST_SEED),
    engine_id: str = "stable-diffusion-xl-1024-v1-0",
) -> List[str]:

    data = {
        # "init_image": open(init_image_path, "rb"),
        "init_image": base64.b64decode(init_image),   # THis might cause issues, may need to use IO or save to file first
        "image_strength": str(image_strength),
        "init_image_mode": init_image_mode,
        "cfg_scale": str(cfg_scale),
        "samples": str(samples),
        "steps": str(steps),
        "seed": str(seed),
    }
    for i, prompt in enumerate(text_prompts):
        data[f"text_prompts[{i}][text]"] = prompt["text"]
        data[f"text_prompts[{i}][weight]"] = str(prompt["weight"])  # Convert weight to string if necessary
    
    if style_preset:
        data["style_preset"] = style_preset

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_HOST}/v1/generation/{engine_id}/image-to-image",
            headers={"Accept": "application/json", "Authorization": f"Bearer {API_KEY}"},
            data=data,
        ) as response:
            image_b64s = []
            if response.status != 200:
                bt.logging.warning("Bad response code :( {} {}".format(response.status, response.reason))

            response_json = await response.json()
            for i, image in enumerate(response_json.get("artifacts", [])):
                image_b64s.append(image["base64"])

        return image_b64s
