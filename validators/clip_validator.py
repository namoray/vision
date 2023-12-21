from __future__ import annotations

import asyncio
import datetime
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import bittensor as bt
import diskcache
import numpy as np
import torch
import clip
import io
from PIL import Image
import base64

from core import utils
from template.protocol import ClipEmbeddingImages, ClipEmbeddingTexts
from validators.base_validator import BaseValidator
from datasets import load_dataset
import markovify



class ClipValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=15)

        self.cache = diskcache.Cache(
            "validator_cache", 
        )
        dataset = load_dataset('multi-train/coco_captions_1107')
        text = [i["query"] for i in dataset["train"]]
        self.markov_text_generation_model = markovify.Text(" ".join(text))


    async def query_miner_with_images(
        self,
        metagraph: bt.metagraph,
        uid: int,
        image_b64s: list[str],
    ) -> Tuple[int, ClipEmbeddingImages]:
        query = ClipEmbeddingImages(
           image_b64s=image_b64s
        )
        return await self.query_miner(metagraph.axons[uid], uid, query)

    async def query_miner_with_texts(
        self,
        metagraph: bt.metagraph,
        uid: int,
        text_prompts: list[str],
    ) -> Tuple[int, ClipEmbeddingTexts]:
        query = ClipEmbeddingTexts(
           text_prompts=text_prompts
        )
        return await self.query_miner(metagraph.axons[uid], uid, query)
    
    def get_expected_image_embeddings(self, image_b64s: list[str]) -> List[List[float]]:
        images = [Image.open(io.BytesIO(base64.b64decode(img_b64))) for img_b64 in image_b64s ]
        images = [self.clip_preprocess(image) for image in images]
        images_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            image_embeddings = self.clip_model.encode_image(images_tensor)
        
        return image_embeddings.cpu().numpy().tolist()
    
    def get_expected_text_embeddings(self, text_prompts: list[str]) -> List[List[float]]:
        texts_tensor = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(texts_tensor)
        
        text_embeddings = text_embeddings.cpu().numpy().tolist()
        return text_embeddings
    
    async def run_image_embedding_query_for_uid(self, uid: int, image_b64s: List[str], metagraph: bt.metagraph) -> Tuple[int, float]:
        random_number_of_images_to_score_on = random.randint(1, 10)
        if len(image_b64s) >= random_number_of_images_to_score_on:
            selected_image_b64s = random.sample(image_b64s, random_number_of_images_to_score_on)
        else:
            selected_image_b64s = image_b64s

        response = await self.query_miner_with_images(metagraph, uid, selected_image_b64s)
        expected_response = self.get_expected_image_embeddings(selected_image_b64s)
        score = self.score_dot_embeddings(expected_response, response[1].image_embeddings)
        return (uid, score)
        

    async def run_text_embedding_query_for_uid(self, uid: int, metagraph: bt.metagraph) -> Tuple[int, float]:
        text_prompts = self.generate_n_random_text_prompts(random.randint(1, 10))
        response = await self.query_miner_with_texts(metagraph, uid, text_prompts)
        expected_response = self.get_expected_text_embeddings(text_prompts)
        score = self.score_dot_embeddings(expected_response, response[1].text_embeddings)
        bt.logging.debug(f"UID: {uid}, EXPECTED_RESPONSE: {expected_response}, RESPONSE: {response[1].text_embeddings}, SCORE: {score}")
        return (uid, score)
    
    async def get_scores_for_image_embeddings(self, image_b64s: list[str], metagraph: bt.metagraph, available_uids: List[int]) -> Dict[int, float]:
        img_tasks = [asyncio.create_task(self.run_image_embedding_query_for_uid(uid, image_b64s, metagraph)) for uid in available_uids]
        uids_and_scores = await asyncio.gather(*img_tasks)
        scores: Dict[int, float] = {}
        for uid, score in uids_and_scores:
            scores[uid] = score
        return scores

    async def get_scores_for_text_embeddings(self, metagraph: bt.metagraph, available_uids: List[int]) -> Dict[int, float]:
        img_tasks = [asyncio.create_task(self.run_text_embedding_query_for_uid(uid, metagraph)) for uid in available_uids]
        uids_and_scores = await asyncio.gather(*img_tasks)
        scores: Dict[int, float] = {}
        for uid, score in uids_and_scores:
            scores[uid] = score
        return scores

    def generate_n_random_text_prompts(self, x: int) -> list[str]:
        return [self.markov_text_generation_model.make_short_sentence(max_chars=200) for _ in range(x)]
    
    @staticmethod
    def score_dot_embeddings(
        expected_embeddings: List[List[float]],
        response_embeddings: List[List[float]],
    ) -> float:
        expected_embeddings = np.array(expected_embeddings)
        response_embeddings = np.array(response_embeddings)
        
        if expected_embeddings.shape != response_embeddings.shape:
            if response_embeddings.shape is not None or len(response_embeddings) == 0:
                bt.logging.warning(f"Expected embeddings shape is {expected_embeddings.shape} but the response embeddings shape is {response_embeddings.shape}")
            return 0
        
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
        return avg
    
