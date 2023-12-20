import argparse
import asyncio
import datetime
import math
import os
import random
import time
import traceback
from typing import Any, Dict, List, Tuple, Union

import bittensor as bt
import torch
import wandb
from core import constants as cst
from core import utils
from template.protocol import IsAlive, SegmentingSynapse, ClipEmbeddingImages, ClipEmbeddingTexts
from validators.segmentation_validator import SegmentationValidator
import template
from validators.clip_validator import ClipValidator

moving_average_scores = torch.zeros(256)
wandb_runs = {}


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=19)
    parser.add_argument("--wandb_off", action="store_false", dest="wandb_on")
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    config.full_path = os.path.expanduser(
        f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator"
    )
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


def init_wandb(config, my_uid, wallet):
    if config.wandb_on:
        run_name = f"validator-{my_uid}-{template.__version__}"
        config.uid = my_uid
        config.hotkey = wallet.hotkey.ss58_address
        config.run_name = run_name
        config.version = template.__version__
        config.type = "validator"

        # Initialize the wandb run for the single project
        run = wandb.init(
            name=run_name,
            project=template.PROJECT_NAME,
            entity="19-vision",
            config=config,
            dir=config.full_path,
            reinit=True,
        )

        # Sign the run to ensure it's from the correct hotkey
        signature = wallet.hotkey.sign(run.id.encode()).hex()
        config.signature = signature
        wandb.config.update(config, allow_val_change=True)

        bt.logging.success(f"Started wandb run for project '{template.PROJECT_NAME}'")


def initialize_components(config):
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}"
    )
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    dendrite = bt.dendrite(wallet=wallet)
    try:
        my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    except ValueError:
        bt.logging.error(
            f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register --netuid 19 and try again."
        )
        exit()
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register --netuid 19 and try again."
        )
        exit()

    return wallet, subtensor, metagraph, dendrite, my_uid


def initialize_validators(vali_config):
    segmentation_vali = SegmentationValidator(**vali_config)
    clip_vali = ClipValidator(**vali_config)
    return segmentation_vali, clip_vali


async def check_uid(dendrite: bt.dendrite, axon: bt.axon, uid: int) -> Union[bt.axon, None]:
    """Asynchronously check if a UID is available."""
    try:
        response = await dendrite.forward(axon, IsAlive(), deserialize=False, timeout=4)
        if response.is_success:
            bt.logging.trace(f"UID {uid} is active")
            return axon  # Return the axon info instead of the UID
        else:
            bt.logging.trace(f"UID {uid} is not active")
            return None
    except Exception as e:
        bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        return None


async def get_available_uids(dendrite: bt.dendrite, metagraph: bt.metagraph) -> Dict[int, bt.axon]:
    """Get a dictionary of available UIDs and their axons asynchronously."""
    tasks = {
        uid.item(): check_uid(dendrite, metagraph.axons[uid.item()], uid.item())
        for uid in metagraph.uids
    }
    results = await asyncio.gather(*tasks.values())

    # Create a dictionary of UID to axon info for active UIDs
    available_uids = {
        uid: axon_info
        for uid, axon_info in zip(tasks.keys(), results)
        if axon_info is not None
    }

    return available_uids


def set_weights(scores: torch.tensor, config: Any, subtensor: bt.subtensor, wallet: bt.wallet, metagraph: bt.metagraph) -> None:
    global moving_average_scores

    alpha = 0.2
    moving_average_scores = alpha * scores + (1 - alpha) * moving_average_scores


    uids_data = metagraph.uids.data
    weights = moving_average_scores[uids_data]
    subtensor.set_weights(
        netuid=config.netuid,
        wallet=wallet,
        uids=metagraph.uids,
        weights=weights,
        wait_for_inclusion=False,
    )
    bt.logging.success("Successfully set weights.")


def update_weights(total_scores: torch.tensor, config: Any, subtensor: bt.subtensor, wallet: bt.wallet, metagraph: bt.metagraph) -> None:
    """Update weights based on total scores, using min-max normalization for display"""

    # Normalize avg_scores to a range of 0 to 1
    min_score = torch.min(total_scores)
    max_score = torch.max(total_scores)

    if max_score - min_score != 0:
        normalized_scores = (total_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = torch.zeros_like(total_scores)
    # We can't set weights with normalized scores because that disrupts the weighting assigned to each validator class
    # Weights get normalized anyways in weight_utils
    bt.logging.info(f"Normalized scores: {normalized_scores}")
    set_weights(total_scores, config, subtensor, wallet, metagraph)


async def get_random_images(uids: Dict[int, bt.axon]) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Returns the unique images with labels, and a dict of which uid gets what image"""
    number_of_images_to_generate = math.ceil(len(uids) / 24)
    tasks = []
    for _ in range(number_of_images_to_generate):
        # These are subject to change, I know what you're thinking, miners.
        x_dim = random.randint(200, 1920)
        y_dim = random.randint(200, 1920)
        tasks.append(asyncio.create_task(utils.get_random_image(x_dim, y_dim)))

    image_b64s = await asyncio.gather(*tasks)

    miners_and_image_b64_labels = {
        uid: random.randint(0, len(image_b64s) - 1) for uid in uids
    }
    images_with_labels = {i: image_b64s[i] for i in range(len(image_b64s))}
    return images_with_labels, miners_and_image_b64_labels


async def score_cache_responses_for_hotkey(
    hotkey: str,
    image_data: Tuple[str, str, datetime.datetime],
    times_to_test: int,
    seg_vali: SegmentationValidator,
    metagraph: bt.metagraph,
    hotkeys_to_uids: Dict[str, int],
):
    """
    Calculate the average score and average time for scoring cache responses for a given hotkey.

    Args:
        hotkey (str): The hotkey to score cache responses for.
        image_data (Tuple[str, str, datetime.datetime]): A tuple containing the image UUID, image base64 data,
            and the timestamp when the image data was captured.
        times_to_test (int): The number of times to test the scoring process.
        seg_vali (SegmentationValidator)
        metagraph (bt.metagraph)
        hotkeys_to_uids (Dict[str, int])

    Returns:
        Tuple[str, float, float]: A tuple containing the hotkey, average score, and average time.
    """
    image_uuid = image_data[0]
    image_base64 = image_data[1]
    image_cv2 = utils.convert_b64_to_cv2_img(image_base64)
    y_dim, x_dim = image_cv2.shape[:2]
    scores = []
    times = []
    uid = hotkeys_to_uids[hotkey]
    bt.logging.info(
        f"Scoring hotkey {hotkey} for uid {uid}, with image_uuid {image_uuid}"
    )
    for _ in range(times_to_test):
        input_boxes, input_points, input_labels = utils.generate_random_inputs(
            x_dim=x_dim, y_dim=y_dim
        )
        time_before = time.time()
        _, response_synapse = await seg_vali.query_miner_with_uuid(
            metagraph,
            uid,
            image_uuid,
            input_boxes,
            input_points,
            input_labels,
        )
        time_taken = time.time() - time_before
        expected_masks = seg_vali._get_expected_json_rle_encoded_masks(
            image_b64=image_base64,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
        )
        score = seg_vali.score_response(response_synapse, expected_masks)
        scores.append(score)
        times.append(time_taken)
        await asyncio.sleep(random.random() * 3)

    average_score = sum(scores) / len(scores)
    average_time = sum(times) / len(times)
    return hotkey, average_score, average_time



async def query_and_score_miners(
    dendrite: bt.dendrite,
    subtensor: bt.subtensor,
    metagraph: bt.metagraph,
    config: Dict[Any, Any],
    wallet: bt.wallet,
    validators: Tuple[SegmentationValidator],
):
    segmenting_vali: SegmentationValidator = validators[0]
    clip_vali: ClipValidator = validators[1]
    hotkeys_to_uids = utils.get_hotkeys_to_uids(metagraph)
    uids_to_hotkeys = utils.get_uids_to_hotkeys(metagraph)
    while True:
        try:
            total_scores = torch.zeros(256)
            metagraph = subtensor.metagraph(config.netuid)
            available_uids = await get_available_uids(dendrite, metagraph)


            ############ SCORING WITH THE CACHE ############
            images_with_labels, miners_and_image_b64_labels = await get_random_images(
                uids=available_uids
            )

            bt.logging.info(
                f"Scoring miners with image b64 now! We have {len(images_with_labels)} images to score, for {len(miners_and_image_b64_labels)} miners"
            )
            (
                segmentation_scores,
                miner_uids_to_image_uuid,
            ) = await segmenting_vali.score_miners_no_image_uuid(
                metagraph, images_with_labels, miners_and_image_b64_labels
            )

            miner_hotkey_to_image_uuid = {
                uids_to_hotkeys[uid]: image_uuid
                for uid, image_uuid in miner_uids_to_image_uuid.items()
            }
            bt.logging.info(f"\nscores from non cache part: {segmentation_scores} \n")
            total_scores = utils.update_total_scores(total_scores, segmentation_scores)

            ############ SCORING WITHOUT THE CACHE ############

            miner_hotkeys_to_image_uuid_and_image = (
                segmenting_vali.update_and_clear_and_fetch_uuid_from_cache(
                    miner_hotkey_to_image_uuid,
                    images_with_labels,
                    miners_and_image_b64_labels,
                    hotkeys_to_uids,
                )
            )

            amount_of_times_to_test_each_hotkey = random.choices(
                cst.POSSIBLE_VALUES_TO_TEST_EACH_HOTKEY,
                cst.WEIGHTS_FOR_NUMBER_OF_TIMES_TO_TEST_EACH_HOTKEY,
                k=1,
            )[0]

            tasks = [
                score_cache_responses_for_hotkey(
                    hotkey,
                    miner_hotkeys_to_image_uuid_and_image[hotkey],
                    amount_of_times_to_test_each_hotkey,
                    segmenting_vali,
                    metagraph,
                    hotkeys_to_uids,
                )
                for hotkey in miner_hotkeys_to_image_uuid_and_image
            ]
            average_scores_and_times = await asyncio.gather(*tasks)
            time_weighted_scores = utils.calculate_time_weighted_scores(
                average_scores_and_times
            )

            scores: Dict[int, float] = {}
            for hotkey, time_weighted_score in time_weighted_scores:
                uid = hotkeys_to_uids[hotkey]
                scores[uid] = time_weighted_score

            bt.logging.info(f"\nscores from cache part: {scores} \n")

            total_scores = utils.update_total_scores(total_scores, scores)

            ############ SCORING IMAGE EMBEDDINGS ############
            
            image_b64s = list(images_with_labels.values())


            scores = {}
            for uid in available_uids:
                random_number_of_images_to_score_on = random.randint(1, 10)
                if len(image_b64s) >= random_number_of_images_to_score_on:
                    selected_image_b64s = random.sample(image_b64s, random_number_of_images_to_score_on)
                else:
                    selected_image_b64s = image_b64s
                
                response = await clip_vali.query_miner_with_images(metagraph, uid, selected_image_b64s)
                expected_response = clip_vali.get_expected_image_embeddings(selected_image_b64s)
                score = clip_vali.score_dot_embeddings(expected_response, response[1].image_embeddings)
                bt.logging.info(f"Image embeddings similarity score for uid {uid}: {score}")
                scores[uid] = score
                
            total_scores = utils.update_total_scores(total_scores, scores)

            ############ SCORING TEXT EMBEDDINGS ############



            bt.logging.info(f"Updating weights !")
            update_weights(
                total_scores, config, subtensor, wallet, metagraph
            )

            bt.logging.info("Bout to sleep for a bit, done scoring for now :)")
            await asyncio.sleep(random.random() *  10)

        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)


def main():
    global validators
    config = get_config()
    wallet, subtensor, metagraph, dendrite, my_uid = initialize_components(config)
    validator_config = {
        "dendrite": dendrite,
        "config": config,
        "subtensor": subtensor,
        "wallet": wallet,
    }
    init_wandb(config, my_uid, wallet)
    validators = initialize_validators(validator_config)
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(
            query_and_score_miners(
                dendrite, subtensor, metagraph, config, wallet, validators
            )
        )

    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

    finally:
        loop.close()


if __name__ == "__main__":
    main()
