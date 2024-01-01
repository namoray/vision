import base64
import hashlib
import io
import uuid
from io import BytesIO
from typing import Dict, List, Tuple

import aiohttp
import bittensor as bt
import cv2
import diskcache
import numpy as np
import requests
import torch
from PIL import Image
import random


async def get_random_image(x_dim: int, y_dim: int) -> str:
    """
    Generate a random image with the specified dimensions, by calling unsplash api.

    Args:
        x_dim (int): The width of the image.
        y_dim (int): The height of the image.

    Returns:
        str: The base64 encoded representation of the generated image.
    """
    async with aiohttp.ClientSession() as session:
        url = f"https://source.unsplash.com/random/{x_dim}x{y_dim}"
        async with session.get(url) as resp:
            data = await resp.read()

    img = Image.open(BytesIO(data))

    buffered = BytesIO()

    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return img_b64


def convert_b64_to_cv2_img(image_b64: str) -> np.ndarray:
    """
    Convert a base64 encoded image to a numpy array representing an image in RGB format.

    Args:
        image_b64 (str): The base64 encoded image.

    Returns:
        np.ndarray: The numpy array representing the image in RGB format.
    """
    img_bytes = base64.b64decode(image_b64)
    img_arr = np.frombuffer(img_bytes, np.uint8)  # Convert byte array to numpy array
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)  # Decode into an image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_image_uuid(image_b64_str: str) -> str:
    """
    Generates a unique UUID for an image based on its base64-encoded string representation.

    Args:
        image_b64_str (str): The base64-encoded string representation of the image.

    Returns:
        str: The unique UUID generated for the image.
    """

    image_bytes = base64.b64decode(image_b64_str)
    image_hash = hashlib.md5(image_bytes)
    image_uuid = uuid.UUID(image_hash.hexdigest())

    return str(image_uuid)


def store_image(encoded_image: str, cache: diskcache.Cache):
    """
    Store an image in the cache.

    Args:
        encoded_image (str): The encoded image as a string.
        cache (diskcache.Cache): The cache to store the image in.

    Returns:
        str: The UUID of the stored image.
    """
    image_bytes = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_bytes))
    image_uuid = get_image_uuid(image_bytes)

    if str(image_uuid) in cache:
        raise ValueError(f"Image with UUID {image_uuid} already in cache.")

    else:
        cache.add(str(image_uuid), image)

    return image_uuid


def rle_encode(mask: np.ndarray) -> List[List[int]]:
    """
    Encode a binary mask using run-length encoding.

    Parameters:
        mask (np.ndarray): The binary mask to be encoded.

    Returns:
        List[List[int]]: A list of runs, where each run is represented by a
        pair of integers [start, length].
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return [[runs[i], runs[i + 1]] for i in range(0, len(runs), 2)]


def rle_decode(mask_rle_pairs: List[List[int]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Decodes a run-length encoded (RLE) mask into a binary image.

    Args:
        mask_rle_pairs (List[List[int]]): A list of pairs representing the run-length encoding of the mask. Each pair contains the starting position and length of a run.
        shape (Tuple[int, int]): The shape of the output binary image.

    Returns:
        np.ndarray: The decoded binary image.

    Note:
        - The mask_rle_pairs list must contain valid pairs of integers.
        - The shape tuple must contain two positive integers.

    Raises:
        None.
    """
    starts, lengths = zip(*mask_rle_pairs)

    starts, lengths = np.array(starts), np.array(lengths)
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode_masks(masks: List[np.ndarray]) -> List[List[List[int]]]:
    """
    Generate a run-length encoding (RLE) for a list of masks.

    Args:
        masks (List[np.ndarray]): A list of masks represented as NumPy arrays.

    Returns:
        List[List[List[int]]]: A list of RLE encodings for each mask.
    """
    rles = [rle_encode(mask) for mask in masks]
    return [i for i in rles if i != []]


def rle_decode_masks(rles: List[List[List[int]]], shape: Tuple[int, int]):
    """
    Decodes a list of run-length encoded masks into a list of binary masks.

    Parameters:
        rles (List[List[List[int]]]): A list of run-length encoded masks.
        shape (Tuple[int, int]): The shape of the masks.

    Returns:
        List[np.ndarray]: A list of binary masks.
    """
    imgs = [rle_decode(rle, shape) for rle in rles]
    return imgs


def update_total_scores(total_scores: torch.tensor, scores: Dict[int, float], weight=1) -> torch.tensor:
    """
    Updates the total scores by adding the given scores to the existing total scores.

    Args:
        total_scores (torch.tensor): The tensor containing the current total scores.
        metagraph (bt.metagraph): The metagraph
        scores (Dict[int, float]): A dictionary mapping user IDs to their segmentation scores.

    Returns:
        torch.tensor: The updated total scores tensor.
    """

    for uid, score in scores.items():
        total_scores[uid] += score * weight
    return total_scores


def get_uids_to_hotkeys(metagraph: bt.metagraph) -> Dict[str, str]:
    """
    Generate a dictionary mapping uids to hotkeys.

    Args:
        metagraph (bt.metagraph): The metagraph

    Returns:
        Dict[str, str]: A dictionary mapping uids to hotkeys.
    """
    uid_to_hotkey = {}
    for uid in metagraph.uids:
        uid_to_hotkey[uid.item()] = metagraph.hotkeys[uid.item()]
    return uid_to_hotkey


def get_hotkeys_to_uids(metagraph: bt.metagraph) -> Dict[str, str]:
    """
    Return a dictionary mapping hotkeys to uids.

    Args:
        metagraph (bt.metagraph): The metagraph

    Returns:
        dict: A dictionary mapping hotkeys (str) to uids (str).
    """
    hotkey_to_uid = {}
    for uid in metagraph.uids:
        hotkey_to_uid[metagraph.hotkeys[uid.item()]] = uid.item()
    return hotkey_to_uid


def generate_random_inputs(x_dim: int, y_dim: int) -> Tuple[List, List, List]:
    """
    Generate random inputs for given dimensions.

    Parameters:
        x_dim (int): The maximum value for the x-dimension.
        y_dim (int): The maximum value for the y-dimension.

    Returns:
        tuple: A tuple containing three elements:
            - input_boxes (list): A list of input boxes. Each box is represented by a list of four coordinates [x1, y1, x2, y2].
            - input_points (list): A list of input points. Each point is represented by a list of two coordinates [x, y].
            - input_labels (list): A list of input labels, either 0 or 1.
    """
    box_prob = np.random.rand()
    if box_prob <= 0.60:
        number_of_input_boxes = 0
    elif box_prob <= 0.95:
        number_of_input_boxes = 1
    else:
        number_of_input_boxes = np.random.randint(2, 11)

    if number_of_input_boxes == 0:
        input_boxes = []
    elif number_of_input_boxes == 1:
        x1 = round(np.random.uniform(0, x_dim), 2)
        y1 = round(np.random.uniform(0, y_dim), 2)
        x2 = round(np.random.uniform(x1, x_dim), 2)
        y2 = round(np.random.uniform(y1, y_dim), 2)
        input_boxes = [x1, y1, x2, y2]
    else:
        input_boxes = []
        for _ in range(number_of_input_boxes):
            x1 = round(np.random.uniform(0, x_dim), 2)
            y1 = round(np.random.uniform(0, y_dim), 2)
            x2 = round(np.random.uniform(x1, x_dim), 2)
            y2 = round(np.random.uniform(y1, y_dim), 2)
            input_boxes.append([x1, y1, x2, y2])

    if number_of_input_boxes <= 1:
        probs = [1 / 1.2**i for i in range(1, 26)]
        probs = [p / sum(probs) for p in probs]
        length = np.random.choice(range(1, 26), p=probs)
        input_points = [
            [
                round(np.random.uniform(0, x_dim), 2),
                round(np.random.uniform(0, y_dim), 2),
            ]
            for _ in range(length)
        ]
        input_labels = np.random.choice([0, 1], size=length).tolist()
    else:
        input_points = None
        input_labels = None

    return input_boxes, input_points, input_labels


def calculate_time_weighted_scores(scores_and_times: List[Tuple[str, float, float]]):
    """
    Function that returns time weighted scores based on the input task scores and times.

    Returns:
        list: A list of tuples containing each hotkey and its corresponding time-weighted score.
    """

    scores_and_times.sort(key=lambda x: x[2])
    if len(scores_and_times) == 0:
        return []
    if len(scores_and_times) == 1:
        weights = [1]
    else:
        weight_increment = (1.25 - 0.75) / (len(scores_and_times) - 1)
        weights = [(0.75 + round(weight_increment * i, 2)) for i in range(len(scores_and_times))]
    time_weighted_scores = [
        (hotkey, avg_score * weights[i]) for i, (hotkey, avg_score, avg_time) in enumerate(scores_and_times)
    ]

    return time_weighted_scores


def send_discord_alert(message: str, webhook_url: str) -> None:
    """
    Send a Discord alert message using a webhook URL.

    Args:
        message (str): The message to be sent as the alert.
        webhook_url (str): The URL of the webhook to send the alert to.
    """

    data = {"content": f"@everyone {message}", "username": "Subnet18 Updates"}
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Discord alert sent successfully!")
        else:
            print(f"Failed to send Discord alert. Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send Discord alert: {e}", exc_info=True)


def generate_random_weight():
    """
    Generate a random weight.

    Returns:
        float: The randomly generated weight.
    """
    if random.random() < 0.5:
        return 1.0
    else:
        dp_case = random.choices(population=[0, 1, 2], weights=[0.1, 0.1, 0.3], k=1)[0]

        number = random.uniform(0.8, 1.2)
        return round(number, dp_case)
