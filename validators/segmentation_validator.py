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

from core import utils
from template.protocol import SegmentingSynapse
from validators.base_validator import BaseValidator


class SegmentationValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=15)

        self.cache = diskcache.Cache(
            "validator_cache", 
        )

    def _get_expected_json_rle_encoded_masks(
        self,
        image_b64: Optional[str] = None,
        input_boxes: Optional[List] = None,
        input_points: Optional[List] = None,
        input_labels: Optional[List] = None,
    ) -> str:
        """
        Does the expected segmentation for later comparison.
        """
        if image_b64 is None:
            raise ValueError("For some reason you didn't supply aa imageb64 please fix this")

        image_cv2 = utils.convert_b64_to_cv2_img(image_b64)

        with threading.Lock():
            self.predictor.set_image(image_cv2)
            if input_boxes is None or len(input_boxes) == 0 or isinstance(input_boxes[0], int) or len(input_boxes) == 1:
                input_points = np.array(input_points) if input_points else None
                input_labels = np.array(input_labels) if input_labels else None
                input_boxes = np.array(input_boxes).squeeze() if input_boxes else None

                all_masks, scores, _ = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_boxes,
                    multimask_output=True,
                )
            else:
                input_boxes_tensor = torch.tensor(input_boxes, device=self.predictor.device)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes_tensor, image_cv2.shape[:2])
                all_masks, scores, logits = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=True,
                )
                all_masks = all_masks.cpu().numpy()
                scores = scores.cpu().numpy()

        if len(all_masks.shape) == 4:
            best_options_indices = np.argmax(scores, axis=1)
            best_masks = all_masks[np.arange(all_masks.shape[0]), best_options_indices, :, :]
        else:
            best_score = np.argmax(scores)
            best_masks = [all_masks[best_score, :, :]]
        encoded_masks = utils.rle_encode_masks(best_masks)

        return encoded_masks

    def update_and_clear_and_fetch_uuid_from_cache(
        self,
        miner_hotkey_to_image_uuids: Dict[str, str],
        images_with_labels: Dict[int, str],
        miners_and_image_b64_labels: Dict[int, int],
        hotkeys_to_uids: Dict[str, int],
    ) -> Dict[str, Tuple[str, str, datetime.datetime]]:
        miner_hotkeys_to_image_uuid_and_image: Dict[str, Tuple[str, str, datetime.datetime]] = {}
        for hotkey in miner_hotkey_to_image_uuids:
            current_stored_image_stuff: list[tuple[str, str, datetime.datetime]] = self.cache.get(hotkey, [])

            # Get rid of stale image_stuff, cya
            updated_image_uuids = [
                (uuid, image_base64, timestamp)
                for uuid, image_base64, timestamp in current_stored_image_stuff
                if datetime.datetime.now() - timestamp < datetime.timedelta(hours=1)
                and uuid is not None
                and image_base64 is not None
            ]

            image_uuid = miner_hotkey_to_image_uuids[hotkey]
            uid = hotkeys_to_uids[hotkey]
            image_base64 = images_with_labels[miners_and_image_b64_labels[uid]]

            updated_image_uuids.append((image_uuid, image_base64, datetime.datetime.now()))
            self.cache[hotkey] = updated_image_uuids

            random_uuid_base64_date = random.choice(updated_image_uuids)
            miner_hotkeys_to_image_uuid_and_image[hotkey] = random_uuid_base64_date

            random_uuid = random_uuid_base64_date[0]

        return miner_hotkeys_to_image_uuid_and_image

    async def query_miner_with_uuid(
        self,
        metagraph: bt.metagraph,
        uid: int,
        image_uuid: str,
        input_boxes: Union[List[float], List[List[float]]],
        input_points: List[List[float]],
        input_labels: List[int],
    ):
        query = SegmentingSynapse(
            image_uuid=image_uuid,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
        )
        return await self.query_miner(metagraph.axons[uid], uid, query)

    async def query_miner_with_image_b64(
        self,
        metagraph: bt.metagraph,
        uid: int,
        image_b64: str,
        input_boxes: Union[List[float], List[List[float]]],
        input_points: List[List[float]],
        input_labels: List[int],
    ):
        query = SegmentingSynapse(
            image_b64=image_b64,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
        )
        return await self.query_miner(metagraph.axons[uid], uid, query)

    def score_response(self, response_synapse: SegmentingSynapse, expected_masks: List[List[List[int]]]) -> float:
        image_shape = response_synapse.image_shape
        response_masks = response_synapse.masks
        if image_shape is None or response_masks is None:
            return 0
        score = self.masks_score_dot(expected_masks, response_masks, shape=image_shape)
        return score

    async def process_image_label(
        self,
        miners_and_image_labels: Dict[int, int],
        image_label: str,
        image_b64: str,
        scores: Dict[int, str],
        uid_to_image_uuid: Dict[str, str],
        metagraph: bt.metagraph,
    ) -> None:
        """For each image, get all the random points we need, get the expected masks, and then query the miners

        Weight their scores based on the relative response times compared to other miners
        """
        image_cv2 = utils.convert_b64_to_cv2_img(image_b64)
        y_dim, x_dim = image_cv2.shape[:2]
        input_boxes, input_points, input_labels = utils.generate_random_inputs(x_dim, y_dim)
        expected_masks = self._get_expected_json_rle_encoded_masks(image_b64, input_boxes, input_points, input_labels)
        if len(expected_masks) == 0:
            bt.logging.error(
                f"JUST GOT 0 EXPECTED MASKS IDK WHY, length_b64={len(image_b64)}, input_boxes={input_boxes}, input_points={input_points}, input_labels={input_labels}"
            )

        query_miners_tasks = [
            self.process_query(
                uid,
                image_b64,
                input_boxes,
                input_points,
                input_labels,
                expected_masks,
                metagraph,
            )
            for uid, image_id in miners_and_image_labels.items()
            if image_label == image_id
        ]

        query_miner_results = await asyncio.gather(*query_miners_tasks)
        # Now extract the image_uuids out for later caching testing
        info_for_calculating_time_weighted_scores = []
        for result in query_miner_results:
            uid, score, time_taken, image_uuid = result
            info_for_calculating_time_weighted_scores.append((uid, score, time_taken))
            uid_to_image_uuid[uid] = image_uuid

        time_weighted_scores = utils.calculate_time_weighted_scores(info_for_calculating_time_weighted_scores)

        for uid, time_weighted_score in time_weighted_scores:
            scores[uid] = time_weighted_score

    async def process_query(
        self,
        uid: int,
        image_b64: Optional[str],
        input_boxes: Optional[Union[List[List[int]], List[int]]],
        input_points: List,
        input_labels: List,
        expected_masks: List,
        metagraph: bt.metagraph,
    ) -> Tuple:
        
        time_before = time.time()
        _, response_synapse = await self.query_miner_with_image_b64(
            metagraph,
            uid,
            image_b64,
            input_boxes,
            input_points,
            input_labels,
        )
        time_taken = time.time() - time_before

        score = self.score_response(response_synapse, expected_masks)
        return uid, score, time_taken, response_synapse.image_uuid

    async def score_miners_no_image_uuid(
        self,
        metagraph: bt.metagraph,
        images_with_labels: Dict[int, str],
        miners_and_image_labels: Dict[int, str],
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Calculates the scores of miners with a image b64 instead of an image_uuid.

        Args:
            metagraph (bt.metagraph):
            images_with_labels (Dict[int, str]):
            miners_and_image_labels (Dict[int, str]):

        Returns:
            Tuple[Dict[str, float], Dict[str, str]]: A tuple containing two dictionaries:
                - scores: A dictionary mapping miner IDs to their corresponding scores.
                - miner_uids_to_image_uuid: A dictionary mapping miner UUIDs to their corresponding image UUIDs (that they just generated).
        """

        miner_uids_to_image_uuid: Dict[str, str] = {}
        scores: Dict[int, float] = {}
        process_miners_for_image_labels = [
            asyncio.create_task(
                self.process_image_label(
                    miners_and_image_labels,
                    image_label,
                    image_b64,
                    scores,
                    miner_uids_to_image_uuid,
                    metagraph,
                )
            )
            for image_label, image_b64 in images_with_labels.items()
        ]

        await asyncio.gather(*process_miners_for_image_labels)
        return scores, miner_uids_to_image_uuid

    @staticmethod
    def masks_score_dot(
        expected_masks_encoded: np.ndarray,
        response_masks_encoded: Union[np.ndarray, List[Any]],
        shape: list[int],
    ):
        expected_masks = utils.rle_decode_masks(expected_masks_encoded, shape)
        response_masks = utils.rle_decode_masks(response_masks_encoded, shape)

        if len(expected_masks) != len(response_masks):
            if len(expected_masks) == 0:
                bt.logging.error(f"Why do we have 0 expected masks? Tis probably an error, please look into it")
            return 0

        cosine_similarities = []
        for i in range(len(expected_masks)):
        
            if np.count_nonzero(expected_masks[i]) == 0 and np.count_nonzero(response_masks[i]) == 0:
                cos_sim = 1
            elif np.count_nonzero(expected_masks[i]) == 0 or np.count_nonzero(response_masks[i]) == 0:
                cos_sim = 0
            else:
                dot_product = np.dot(
                    expected_masks[i].flatten().astype(float),
                    response_masks[i].flatten().astype(float),
                )
                cos_sim = dot_product / (np.linalg.norm(expected_masks[i]) * np.linalg.norm(response_masks[i]))
        
            cosine_similarities.append(cos_sim)

        if len(cosine_similarities) == 0:
            avg = 0
        else:
            avg = sum(cosine_similarities) / len(cosine_similarities)
        return avg
