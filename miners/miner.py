import argparse
import asyncio
import base64
import copy
import io
import threading
import time
import traceback
from typing import Any, Dict, Tuple, TypeVar

import bittensor as bt
import clip
import diskcache
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from core import constants as cst
from core import utils
from miners.config import check_config, get_config
from template import protocol
from core import stability_api

# import base miner class which takes care of most of the boilerplate


T = TypeVar("T", bound=bt.Synapse)


class MinerBoi:
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)
        check_config(MinerBoi, self.config)
        self.prompt_cache: dict[str, tuple[str, int]] = {}
        self.request_timestamps = {}

        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info("Setting up bittensor objects.")

        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"Wallet {self.wallet}")

        bt.logging.info(f"Config: {self.config}")

        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint}")

        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your miner / validator in the wallet: {self.wallet}, is not registered to this subnet on chain connection: {self.subtensor}. Run btcli register and try again. "
            )
            exit()
        else:
            self.my_hotkey_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running miner on uid: {self.my_hotkey_uid}")

        bt.logging.info("Starting Segmenting miner")

        self.device = self.config.neuron.device
        bt.logging.debug(f"Using device: {self.device} on the miner")

        sam = sam_model_registry[cst.MODEL_TYPE](checkpoint=cst.CHECKPOINT_PATH)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.axon = axon or bt.axon(wallet=self.wallet, port=self.config.axon.port)

        bt.logging.info(f"Attaching Embeddings and Segmenting functions to axon.")

        self.axon.attach(
            forward_fn=self.get_segmentation,
            blacklist_fn=self.blacklist_segmentation,
        ).attach(
            forward_fn=self._is_alive,
            blacklist_fn=self.blacklist_is_alive,
        ).attach(
            forward_fn=self.get_image_embeddings,
            blacklist_fn=self.blacklist_image_embeddings,
        ).attach(
            forward_fn=self.get_text_embeddings,
            blacklist_fn=self.blacklist_text_embeddings,
        ).attach(
            forward_fn=self.generate_images_from_text,
            blacklist_fn=self.blacklist_generate_images_from_text,
        ).attach(
            forward_fn=self.generate_images_from_image,
            blacklist_fn=self.blacklist_generate_images_from_image,
        )

        bt.logging.info(
            f"Serving attached axons on network:"
            f"{self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.asyncio_lock = asyncio.Lock()
        self.threading_lock = threading.Lock()
        self.request_timestamps: Dict[Any, Any] = {}

        self.cache = diskcache.Cache("images_cache", size_limit=cst.MINER_CACHE_SIZE)

    def _is_alive(self, synapse: protocol.IsAlive) -> protocol.IsAlive:
        bt.logging.info("I'm alive!")
        synapse.answer = "alive"
        return synapse

    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        return bt.config(parser)

    async def generate_images_from_text(
        self, synapse: protocol.GenerateImagesFromText
    ) -> protocol.GenerateImagesFromText:
        image_b64s = await stability_api.generate_images_from_text(
            text_prompts=synapse.text_prompts,
            cfg_scale=synapse.cfg_scale,
            height=synapse.height,
            width=synapse.width,
            samples=synapse.samples,
            steps=synapse.steps,
            style_preset=synapse.style_preset,
            seed=synapse.seed,
        )

        synapse.image_b64s = image_b64s
        return synapse

    async def generate_images_from_image(
        self, synapse: protocol.GenerateImagesFromImage
    ) -> protocol.GenerateImagesFromImage:

        bt.logging.debug(f"Here and about to generate an image")
        
        image_b64s = await stability_api.generate_images_from_image(
            init_image=synapse.init_image,
            text_prompts=synapse.text_prompts,
            cfg_scale=synapse.cfg_scale,
            samples=synapse.samples,
            sampler=synapse.sampler,
            steps=synapse.steps,
            init_image_mode=synapse.init_image_mode,
            image_strength=synapse.image_strength,
            style_preset=synapse.style_preset,
            seed=synapse.seed,
        )

        # Remove to minimise data transferred
        synapse.init_image = None
        synapse.image_b64s = image_b64s

        return synapse

    async def get_segmentation(self, synapse: protocol.SegmentingSynapse) -> protocol.SegmentingSynapse:
        """
        Generates the masks for an image, points, labels & boxes. This function is the core.
        You know you love it.

        Parameters:
        - synapse (protocol.SegmentingSynapse)

        Returns:
        - synapse (protocol.SegmentingSynapse): Now with el masks in there
        """
        bt.logging.debug(f"Gonna generate some masks like the good little miner I am")

        if synapse.image_uuid is None and synapse.image_b64 is None:
            synapse.error_message = "❌ You must supply an image or UUID of the already stored image"
            bt.logging.warning(f"USER ERROR: {synapse.error_message}, synapse: {synapse}")
            return synapse

        if synapse.image_uuid is not None and synapse.image_uuid in self.cache:
            image_cv2 = self.cache[synapse.image_uuid]
        elif synapse.image_b64 is not None:
            try:
                image_uuid = utils.get_image_uuid(synapse.image_b64)
            except Exception:
                synapse.error_message = "❌ Failed to get image uuid form image base64, invalid base64 I think"
                bt.logging.error(f" USER ERROR: {synapse.error_message}")
                return synapse
            image_cv2 = utils.convert_b64_to_cv2_img(synapse.image_b64)
            bt.logging.info("Image not found in cache, gonsta store it now")
            self.cache.set(image_uuid, image_cv2)
            synapse.image_uuid = image_uuid

        else:
            synapse.error_message = (
                "❌ Image not found in cache and you didn't supply the image :( Can you please gimme the image?!)"
            )
            bt.logging.warning(f" USER ERROR: {synapse.error_message}")
            return synapse

        # remove image from synapse to not transfer it all back over the web again. Smort.
        synapse.image_b64 = None

        if synapse.input_points is None and synapse.input_boxes is None and synapse.input_labels is None:
            synapse.error_message = "❌ No input points, boxes or labels, just gonna store the image"
            bt.logging.warning(f" USER ERROR: {synapse.error_message}")
            return synapse

        async with self.asyncio_lock:
            self.predictor.set_image(image_cv2)
            if (
                synapse.input_boxes is None
                or len(synapse.input_boxes) == 0
                or isinstance(synapse.input_boxes[0], int)
                or len(synapse.input_boxes) == 1
            ):
                input_points = np.array(synapse.input_points) if synapse.input_points else None

                input_labels = np.array(synapse.input_labels) if synapse.input_labels else None
                input_boxes = np.array(synapse.input_boxes).squeeze() if synapse.input_boxes else None

                all_masks, scores, _ = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_boxes,
                    multimask_output=True,
                )
            else:
                input_boxes_tensor = torch.tensor(synapse.input_boxes, device=self.predictor.device)
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
        synapse.masks = encoded_masks
        synapse.image_shape = list(image_cv2.shape)[:2]

        if len(encoded_masks) > 0:
            bt.logging.info(f"✅ Generated {len(synapse.masks)} mask(s), go me")

        return synapse

    async def get_image_embeddings(self, synapse: protocol.ClipEmbeddingImages) -> protocol.ClipEmbeddingImages:
        if synapse.image_b64s is None:
            synapse.error_message = "❌ You must supply the images that you want to embed"
            bt.logging.warning(f"USER ERROR: {synapse.error_message}, synapse: {synapse}")
            return synapse

        images = [Image.open(io.BytesIO(base64.b64decode(img_b64))) for img_b64 in synapse.image_b64s]
        async with self.asyncio_lock:
            images = [self.clip_preprocess(image) for image in images]
            images_tensor = torch.stack(images).to(self.device)
            with torch.no_grad():
                image_embeddings = self.clip_model.encode_image(images_tensor)

        image_embeddings = image_embeddings.cpu().numpy().tolist()
        synapse.image_embeddings = image_embeddings
        if len(image_embeddings) > 0:
            bt.logging.info(f"✅ {len(synapse.image_embeddings)} image embedding(s) generated. bang.")

        # Removing this to not transfer it all back over the web again.
        synapse.image_b64s = None

        return synapse

    async def get_text_embeddings(self, synapse: protocol.ClipEmbeddingTexts) -> protocol.ClipEmbeddingTexts:
        if synapse.text_prompts is None:
            synapse.error_message = "❌ You must supply the text prompts that you want to embed"
            bt.logging.warning(f"USER ERROR: {synapse.error_message}, synapse: {synapse}")
            return synapse

        text_prompts = synapse.text_prompts

        texts_tensor = clip.tokenize(text_prompts).to(self.device)
        async with self.asyncio_lock:
            with torch.no_grad():
                text_embeddings = self.clip_model.encode_text(texts_tensor)

        list_text_embeddings = text_embeddings.cpu().numpy().tolist()
        synapse.text_embeddings = list_text_embeddings
        bt.logging.info(f"✅ Generated {len(list_text_embeddings)} text embedding(s)? Completed it mate")

        # Removing this to not transfer it all back over the web again.
        synapse.text_prompts = None

        return synapse

    async def blacklist(self, synapse: T) -> Tuple[bool, str]:
        return False, synapse.dendrite.hotkey
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, synapse.dendrite.hotkey

    async def blacklist_is_alive(self, synapse: protocol.IsAlive) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_segmentation(self, synapse: protocol.SegmentingSynapse) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_image_embeddings(self, synapse: protocol.ClipEmbeddingImages) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_text_embeddings(self, synapse: protocol.ClipEmbeddingTexts) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_generate_images_from_text(self, synapse: protocol.GenerateImagesFromText) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_generate_images_from_image(self, synapse: protocol.GenerateImagesFromImage) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority(self, synapse: T) -> float:
        """
        The priority function determines the order in which requests are handled.
        """
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority)
        return priority

    async def priority_isalive(self, synapse: protocol.IsAlive) -> float:
        return await self.priority(synapse)

    async def priority_segmentation(self, synapse: protocol.SegmentingSynapse) -> float:
        return await self.priority(synapse)

    async def priority_image_embeddings(self, synapse: protocol.ClipEmbeddingImages) -> float:
        return await self.priority(synapse)

    async def priority_text_embeddings(self, synapse: protocol.ClipEmbeddingTexts) -> float:
        return await self.priority(synapse)

    async def priority_generate_images_from_text(self, synapse: protocol.GenerateImagesFromText) -> float:
        return await self.priority(synapse)

    async def priority_generate_images_from_image(self, synapse: protocol.GenerateImagesFromImage) -> float:
        return await self.priority(synapse)

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
                f"Please register the hotkey using `btcli s register --netuid 18` before trying again"
            )
            exit()
        bt.logging.info(
            f"Serving axon {self.axon.ip} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info(f"Starting main loop")
        step = 0
        try:
            while not self.should_exit:
                start_epoch = time.time()

                # --- Wait until next epoch.
                current_block = self.subtensor.get_current_block()
                while current_block - self.last_epoch_block < self.config.miner.blocks_per_epoch:
                    # --- Wait for next bloc.
                    time.sleep(1)
                    current_block = self.subtensor.get_current_block()
                    # --- Check if we should exit.
                    if self.should_exit:
                        break

                # --- Update the metagraph with the latest network state.
                self.last_epoch_block = self.subtensor.get_current_block()

                self.metagraph = self.subtensor.metagraph(
                    netuid=self.config.netuid,
                    lite=True,
                    block=self.last_epoch_block,
                )
                log = (
                    f"Step:{step} | "
                    f"Block:{self.metagraph.block.item()} | "
                    f"Stake:{self.metagraph.S[self.my_hotkey_uid]} | "
                    f"Rank:{self.metagraph.R[self.my_hotkey_uid]} | "
                    f"Trust:{self.metagraph.T[self.my_hotkey_uid]} | "
                    f"Consensus:{self.metagraph.C[self.my_hotkey_uid] } | "
                    f"Incentive:{self.metagraph.I[self.my_hotkey_uid]} | "
                    f"Emission:{self.metagraph.E[self.my_hotkey_uid]}"
                )
                bt.logging.info(log)

                # --- Set weights.
                if not self.config.miner.no_set_weights:
                    pass
                step += 1

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()


if __name__ == "__main__":
    with MinerBoi() as miner:
        while True:
            bt.logging.info("Miner running... you can hopefully relax now :)")
            time.sleep(240)
