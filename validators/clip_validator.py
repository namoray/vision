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


class ClipValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=15)

        self.cache = diskcache.Cache(
            "validator_cache", 
        )

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
        texts: list[str],
    ) -> Tuple[int, ClipEmbeddingTexts]:
        query = ClipEmbeddingTexts(
           texts=texts
        )
        return await self.query_miner(metagraph.axons[uid], uid, query)
    
    def get_expected_image_embeddings(self, image_b64s: list[str]) -> List[List[float]]:
        images = [Image.open(io.BytesIO(base64.b64decode(img_b64))) for img_b64 in image_b64s ]
        images = [self.clip_preprocess(image) for image in images]
        images_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            image_embeddings = self.clip_model.encode_image(images_tensor)
        
        return image_embeddings.cpu().numpy().tolist()
    
    def get_expected_text_embeddings(self, texts: list[str]) -> List[List[float]]:
        texts = [self.clip_preprocess(text) for text in texts]
        texts_tensor = torch.stack(texts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(texts_tensor)
        
        text_embeddings = text_embeddings.cpu().numpy().tolist()
        return text_embeddings
    
    @staticmethod
    def score_dot_embeddings(
        expected_embeddings: List[List[float]],
        response_embeddings: List[List[float]],
    ) -> float:
        expected_embeddings = np.array(expected_embeddings)
        response_embeddings = np.array(response_embeddings)
        
        if expected_embeddings.shape != response_embeddings.shape:
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