from __future__ import annotations

import asyncio
import base64
import io
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import diskcache
import bittensor as bt

from validators.base_validator import BaseValidator
import os


class StabilityValidator(BaseValidator):
    def __init__(self, dendrite: bt.dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=5)

        self.stability_api_key = os.environ.get("STABILITY_API_KEY")
        self.dendrite = dendrite
        self.cache = diskcache.Cache(
            "validator_cache/stability_images",
        )
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
