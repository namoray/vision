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
from core import stability_api, utils, dataclasses as dc
from validators.base_validator import BaseValidator
from template import protocol
import os

CFG_SCALE_VALUES = list(range(0, 36))
HEIGHT_VALUES = [i for i in range(320, 1537) if i % 64 == 0]
WIDTH_VALUES = [i for i in range(320, 1537) if i % 64 == 0]
SAMPLES_VALUES = list(range(1, 4))
STEPS_VALUES = list(range(3, 51))
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
        super().__init__(dendrite, config, subtensor, wallet, timeout=25)

        self.dendrite = dendrite
        self.stability_cache = diskcache.Cache("validator_cache/stability_images", max_size=5 * 1024 * 1024 * 1024)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.list_of_cache_keys = self.stability_cache.iterkeys()

    def update_cache_with_images_and_prompts(
        self, image_b64s: list[str], positive_prompt: str, negative_prompt: str
    ) -> None:
        key = str(uuid4())
        self.stability_cache[key] = (image_b64s, positive_prompt, negative_prompt)
        self.list_of_cache_keys.append(key)

    async def query_miner(self, axon: bt.axon, uid: int, syn: bt.Synapse) -> Tuple[int, bt.Synapse]:
        try:
            responses = await self.dendrite(
                [axon],
                syn,
                deserialize=False,
                timeout=self.timeout,
                streaming=self.streaming,
            )
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

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
            new_positive_prompt, new_negative_prompt = self.get_refined_positive_and_negative_prompts(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
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
        markov_text = self.get_markov_text_for_prompt()
        positive_prompt, negative_prompt = await self.get_positive_and_negative_prompt_from_markov_text(markov_text)

        text_prompts = self.get_text_prompt_from_positive_and_negative_prompts(positive_prompt, negative_prompt)
        

        bt.logging.debug(f"Text prompts: {text_prompts}")

        cfg_scale = random.choice(CFG_SCALE_VALUES)
        height = random.choice(HEIGHT_VALUES)
        width = random.choice(WIDTH_VALUES)
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

    async def query_and_score_text_to_image(self, metagraph: bt.metagraph, available_uids: Dict[int, bt.axon]):
        bt.logging.debug(f"Scoring text to images for {len(available_uids)} miners.")

        args = await self.get_args_for_text_to_image()
        bt.logging.debug(f"Args: {args}")

        get_image_task = asyncio.create_task(stability_api.generate_images_from_text(**args))

        query_miners_for_images_tasks = []
        for uid, axon in available_uids.items():
            synapse = protocol.GenerateImagesFromText(**args)
            query_miners_for_images_tasks.append(asyncio.create_task(self.query_miner(axon, uid, synapse)))

        expected_image_b64s = await get_image_task
        positive_prompt = args["text_prompts"][0].text 
        negative_prompt = args["text_prompts"][1].text if len(args["text_prompts"]) >= 1 else ""
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
        bt.logging.debug(f"Args: {args}")

        get_image_task = asyncio.create_task(stability_api.generate_images_from_image(**args))

        query_miners_for_images_tasks = []
        for uid, axon in available_uids.items():
            synapse = protocol.GenerateImagesFromImage(**args)
            query_miners_for_images_tasks.append(asyncio.create_task(self.query_miner(axon, uid, synapse)))

        expected_image_b64s = await get_image_task
        random_image_uuid = str(uuid4())
        self.stability_cache.set(random_image_uuid, expected_image_b64s)

        bt.logging.debug(f"Expecting {len(expected_image_b64s)} image(s) to score")

        results: list[tuple[int, Optional[protocol.GenerateImagesFromImage]]] = await asyncio.gather(
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

    async def get_positive_and_negative_prompt_from_markov_text(self, markov_text: str) -> Tuple[str, str]:
        """Get positive and negative prompt from markov text."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Make the below very similar but making more sense. One sentence. If there's two prompts, ignore one. Make sure it looks like a DESCRIPTION of an image. Try not to change the original text much.",
                },
                {"role": "user", "content": markov_text},
            ],
            temperature=0.2,
        )
        positive_prompt = response.choices[0].message.content

        if random.random() < 0.5:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Take in a description of an image, and then give a few words describing what you don't want to be in the image. Just a few words is good. Seperate each word (or phrase) with a comma. Dont make it sound like a sentence. So just a few words or phrases seperated by commas. Don't use the prefix `no`",
                    },
                    {"role": "user", "content": positive_prompt},
                ],
                temperature=0.2,
            )
            negative_prompt = response.choices[0].message.content
        else:
            negative_prompt = ""

        return positive_prompt, negative_prompt

    async def get_refined_positive_and_negative_prompts(self, positive_prompt: str, negative_prompt: str):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You will be given a prompt for an image. It will include a `positive prompt`, which is things in the image, and a `negative prompt`, which are things not in the image. I want you to just pick one or two things to change about the image, and reflect that in the positive and negative prompts. Don't change it very much, keep it really similar. Return in this format only: \n POSITIVE_PROMPT: blahblah\nNEGATIVE_PROMPT: blahblah",
                },
                {"role": "user", "content": f"POSITIVE_PROMPT: {positive_prompt}\nNEGATIVE_PROMPT: {negative_prompt}"},
            ],
        )
        revised_prompt = response.choices[0].message.content

        positive_prompt_new = revised_prompt.split("POSITIVE_PROMPT: ")[1].split("\n")[0]
        negative_prompt_new = revised_prompt.split("NEGATIVE_PROMPT: ")[1].split("\n")[0]

        if len(positive_prompt_new) == 0:
            positive_prompt_new = positive_prompt
        if len(negative_prompt_new) == 0:
            negative_prompt_new = negative_prompt

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
