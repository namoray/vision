from __future__ import annotations

import asyncio
import base64
import io
import random
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
import numpy as np
import diskcache
import bittensor as bt
from openai import OpenAI
from core import stability_api, utils, dataclasses as dc, constants as cst
from validators.base_validator import BaseValidator
from template import protocol
import os
from dotenv import load_dotenv
from PIL import Image
import markovify

load_dotenv()

CFG_SCALE_VALUES = list(range(0, 36))
HEIGHT_VALUES = [i for i in range(320, 1537) if i % 64 == 0]
WIDTH_VALUES = [i for i in range(320, 1537) if i % 64 == 0]
SAMPLES_VALUES = [1 for _ in range(50)] + [2, 2, 3]
STEPS_VALUES = list(range(10, 51))
STYLE_PRESET_VALUES = [
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
    None,
    None,
    None,
    None,
    None,
]
IMAGE_STRENGTH_VALUES = [i * 0.01 for i in range(0, 100)]
SAMPLER_VALUES = [
    "DDIM",
    "DDPM",
    "K_DPMPP_2M",
    "K_DPMPP_2S_ANCESTRAL",
    "K_DPM_2",
    "K_DPM_2_ANCESTRAL",
    "K_EULER",
    "K_EULER_ANCESTRAL",
    "K_HEUN",
    "K_LMS",
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]


class StabilityValidator(BaseValidator):
    def __init__(self, dendrite: bt.dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=60)

        self.dendrite = dendrite
        self.stability_cache = diskcache.Cache("validator_cache/stability_images", max_size=5 * 1024 * 1024 * 1024)
        self.list_of_cache_keys = list(self.stability_cache.iterkeys())

    def update_cache_with_images_and_prompts(
        self, image_b64s: list[str], positive_prompt: str, negative_prompt: str
    ) -> None:
        key = str(uuid4())
    
        compressed_images_b64s = []
        for img_b64 in image_b64s:
            img_bytes = base64.b64decode(img_b64)
            img_io = io.BytesIO(img_bytes)
            img_pil = Image.open(img_io)
            output = io.BytesIO()
            img_pil.save(output, format="WEBP", quality=80)
            
            img_b64_compressed = base64.b64encode(output.getvalue()).decode()
            
            compressed_images_b64s.append(img_b64_compressed)
        
        self.stability_cache[key] = (compressed_images_b64s, positive_prompt, negative_prompt)
        self.list_of_cache_keys.append(key)


    def get_markov_text_for_prompt(self):
        return self.markov_text_generation_model.make_short_sentence(max_chars=200)

    def get_text_prompt_from_positive_and_negative_prompts(
        self, positive_prompt: str, negative_prompt: str
    ):
        positive_weight = utils.generate_random_weight()
        if random.random() < 0.85:
            text_prompts = [dc.TextPrompt(**{"text": positive_prompt, "weight": positive_weight})]
        else:
            text_prompts = [dc.TextPrompt(**{"text": positive_prompt})]

        if negative_prompt:
            negative_weight = -1.0 * utils.generate_random_weight()
            text_prompts.append(dc.TextPrompt(**{"text": negative_prompt, "weight": negative_weight}))
        
        return text_prompts

    async def get_args_for_image_to_image(self):
        if self.list_of_cache_keys == []:
            return None

        else:
            random_images, positive_prompt, negative_prompt = self.stability_cache.get(
                random.choice(self.list_of_cache_keys)
            )
            init_image = random.choice(random_images)
            new_positive_prompt, new_negative_prompt = await self.get_refined_positive_and_negative_prompts(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
            text_prompts = self.get_text_prompt_from_positive_and_negative_prompts(
                new_positive_prompt, new_negative_prompt
            )

            cfg_scale = random.choice(CFG_SCALE_VALUES)
            samples = random.choice(SAMPLES_VALUES)
            steps = random.choice(STEPS_VALUES)
            sampler = random.choice(SAMPLER_VALUES)
            style_preset = random.choice(STYLE_PRESET_VALUES)
            image_strength = random.choice(IMAGE_STRENGTH_VALUES)

            return {
                "text_prompts": text_prompts,
                "init_image": init_image,
                "cfg_scale": cfg_scale,
                "samples": samples,
                "steps": steps,
                "sampler": sampler,
                "style_preset": style_preset,
                "image_strength": image_strength,
            }

    async def get_args_for_text_to_image(self):
        positive_prompt, negative_prompt = await self.get_positive_and_negative_prompt()

        text_prompts = self.get_text_prompt_from_positive_and_negative_prompts(positive_prompt, negative_prompt)

        bt.logging.debug(f"Text prompts: {text_prompts}")

        cfg_scale = random.choice(CFG_SCALE_VALUES)
        height, width = random.choice(cst.ALLOWED_IMAGE_SIZES)
        samples = random.choice(SAMPLES_VALUES)
        steps = random.choice(STEPS_VALUES)
        sampler = random.choice(SAMPLER_VALUES)
        style_preset = random.choice(STYLE_PRESET_VALUES)


        hyper_parameters = {
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
            "style_preset": style_preset,
        }
        return {"text_prompts": text_prompts, **hyper_parameters}

    async def get_args_for_upscale(self):
        x_dim = random.randint(200, 1920)
        y_dim = random.randint(200, 1920)

        random_image = await utils.get_random_image(x_dim, y_dim)

        base = random.choice([0,1 ])
        max_number = 2048 ** 2 // (x_dim * y_dim)

        if base == 1:
            height = random.randint(base, max_number)
            return {"image": random_image, "height": height}
        else:
            width = random.randint(base, max_number)
            {"image": random_image, "width": width}
        
        
        
    
    async def create_query_img2img_tasks(self, args, available_uids):
        query_miners_for_images_tasks = []
        for uid, axon in available_uids.items():
            synapse = protocol.GenerateImagesFromImage(**args)
            query_miners_for_images_tasks.append(asyncio.create_task(self.query_miner(axon, uid, synapse)))
            bt.logging.debug(f"making new task")
            await asyncio.sleep(1)
        
        return query_miners_for_images_tasks
    
    async def query_and_score_upscale(self, available_uids: Dict[int, bt.axon]):
        bt.logging.debug(f"Scoring upscale for {len(available_uids)} miners.")

        args = await self.get_args_for_upscale()

        query_miners_for_images_tasks = []
        for uid, axon in available_uids.items():
            synapse = protocol.UpscaleImage(**args)
            query_miners_for_images_tasks.append(asyncio.create_task(self.query_miner(axon, uid, synapse)))

        get_image_task = asyncio.create_task(stability_api.upscale_image(**args))

        expected_image_b64s = await get_image_task
        results = await asyncio.gather(*query_miners_for_images_tasks)

        scores = {}
        for uid, response_synapse in results:
            if response_synapse is None or response_synapse.image_b64s is None:
                continue

            score = (
                1
                if response_synapse.image_b64s is not None
                and len(response_synapse.image_b64s) == len(expected_image_b64s)
                else 0
            )
            scores[uid] = score

        bt.logging.info("scores: {}".format(scores))
        return scores


    async def query_and_score_text_to_image(self, metagraph: bt.metagraph, available_uids: Dict[int, bt.axon]):
        bt.logging.debug(f"Scoring text to images for {len(available_uids)} miners.")

        args = await self.get_args_for_text_to_image()
        bt.logging.debug(f"Args: {args}")


        query_miners_for_images_tasks = []
        for uid, axon in available_uids.items():
            synapse = protocol.GenerateImagesFromText(**args)
            query_miners_for_images_tasks.append(asyncio.create_task(self.query_miner(axon, uid, synapse)))

        get_image_task = asyncio.create_task(stability_api.generate_images_from_text(**args))

        expected_image_b64s = await get_image_task
        positive_prompt = args["text_prompts"][0].text 
        negative_prompt = args["text_prompts"][1].text if len(args["text_prompts"]) > 1 else ""
        if len(expected_image_b64s) > 0:
            self.update_cache_with_images_and_prompts(
                expected_image_b64s,  positive_prompt, negative_prompt
            )

        bt.logging.debug(f"Expecting {len(expected_image_b64s)} image(s) to score")

        results: list[tuple[int, Optional[protocol.GenerateImagesFromText]]] = await asyncio.gather(
            *query_miners_for_images_tasks
        )
        scores = {}
        for uid, response_synapse in results:
            if response_synapse is None or response_synapse.image_b64s is None:
                continue

            bt.logging.debug(f"Recieved {len(response_synapse.image_b64s)} images")
            score = (
                1
                if response_synapse.image_b64s is not None
                and len(response_synapse.image_b64s) == len(expected_image_b64s)
                else 0
            )
            scores[uid] = score

        bt.logging.info("scores: {}".format(scores))
        return scores

    async def query_and_score_image_to_image(self, metagraph: bt.metagraph, available_uids: Dict[int, bt.axon]):
        bt.logging.debug(f"Scoring image to image for {len(available_uids)} miners.")

        args = await self.get_args_for_image_to_image()

        if args is None:
            return {}

        get_image_task = asyncio.create_task(stability_api.generate_images_from_image(**args))

        query_miners_for_images_tasks = await self.create_query_img2img_tasks(args, available_uids)

        expected_image_b64s = await get_image_task
        positive_prompt = args["text_prompts"][0].text 
        negative_prompt = args["text_prompts"][1].text if len(args["text_prompts"]) > 1 else ""
        if len(expected_image_b64s) > 0:
            self.update_cache_with_images_and_prompts(
                expected_image_b64s,  positive_prompt, negative_prompt
            )

        results: list[tuple[int, Optional[protocol.GenerateImagesFromImage]]] = await asyncio.gather(
            *query_miners_for_images_tasks
        )
        bt.logging.info("Got all images!")
        scores = {}
        for uid, response_synapse in results:
            if response_synapse is None or response_synapse.image_b64s is None:
                continue

            score = (
                1
                if response_synapse.image_b64s is not None
                and len(response_synapse.image_b64s) == len(expected_image_b64s)
                else 0
            )
            scores[uid] = score

        bt.logging.info("scores: {}".format(scores))
        return scores

    async def get_positive_and_negative_prompt(self) -> Tuple[str, str]:
        """Get positive and negative prompt from markov generated text."""
        positive_prompt = self.markov_text_generation_model.make_sentence(min_words=1, max_words=100)
        
        negative_prompt = self.markov_text_generation_model.make_sentence(min_words=1, max_words=20)

        return positive_prompt, negative_prompt

    def generate_new_prompt(self, prompt):
        words = prompt.split()
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        random_bigram = random.choice(bigrams)
        attemps = 0
        varied_text = prompt
        while attemps < 10:
            try:
                varied_text = self.markov_text_generation_model.make_sentence_with_start(beginning=random_bigram)
                break
            except markovify.text.ParamError: # if bigram doesn't exist, try another one
                random_bigram = random.choice(bigrams)
                attemps += 1
            
        return varied_text

    async def get_refined_positive_and_negative_prompts(self, positive_prompt: str, negative_prompt: str):
        """Get slightly refined positive & negative prompts from markov generated text"""

        positive_prompt_new = self.generate_new_prompt(positive_prompt)
        negative_prompt_new = self.generate_new_prompt(negative_prompt)

        return positive_prompt_new, negative_prompt_new

    @staticmethod
    def score_dot_embeddings(
        expected_embeddings: List[List[float]],
        response_embeddings: List[List[float]],
    ) -> float:
        expected_embeddings = np.array(expected_embeddings)
        response_embeddings = np.array(response_embeddings)

        if expected_embeddings.shape != response_embeddings.shape:
            if expected_embeddings.size == 0:
                bt.logging.error(f"Expected embeddings size is 0. Please check this")
            return 0

        cosine_similarities = []
        for expected, response in zip(expected_embeddings, response_embeddings):
            expected = expected.flatten().astype(float)
            response = response.flatten().astype(float)

            dot_product = np.dot(expected, response)
            cos_sim = dot_product / (np.linalg.norm(expected) * np.linalg.norm(response))
            cosine_similarities.append(cos_sim)

        avg = np.mean(cosine_similarities)
        return round(avg, 2)
