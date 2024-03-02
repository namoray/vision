import base64
import hashlib
import io
import math
import uuid
from io import BytesIO
from typing import List, Tuple
import cv2
import diskcache
import numpy as np
import torch
import imagehash
from PIL import Image, ImageOps
from torch.cuda.amp import autocast
from models import utility_models
from core import constants as cst
from core import dataclasses as dc
from core import resource_management


def crop_images(image_array: List[Image.Image], width: int, height: int) -> List[Image.Image]:
    cropped_images = []
    for image in image_array:
        old_width, old_height = image.size
        if old_width < width or old_height < height:
            cropped_images.append(image)
        else:
            left = (old_width - width) / 2
            top = (old_height - height) / 2
            right = (old_width + width) / 2
            bottom = (old_height + height) / 2

            cropped_image = image.crop((left, top, right, bottom))
            cropped_images.append(cropped_image)
    return cropped_images


def pad_image_pil(img: Image.Image, multiple: int, pad_value: int = 255) -> Image.Image:
    # Calculate the number of rows and columns to be padded
    width, height = img.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    # Pad the image
    # 'fill=255' pads with white for an 8-bit image
    padded_img = ImageOps.expand(
        img,
        (pad_width // 2, pad_height // 2, (pad_width + 1) // 2, (pad_height + 1) // 2),
        fill=pad_value,
    )

    return padded_img


def pad_image_mask_nd(img: np.ndarray, multiple: int, pad_value: int = 255) -> np.ndarray:
    # Calculate the number of rows and columns to be padded
    pad_rows = (multiple - img.shape[0] % multiple) % multiple
    pad_cols = (multiple - img.shape[1] % multiple) % multiple

    # Pad the image
    # 'constant_values=pad_value' pads with the given value
    padded_img = np.pad(
        img,
        ((pad_rows // 2, (pad_rows + 1) // 2), (pad_cols // 2, (pad_cols + 1) // 2)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded_img


def image_is_nsfw(image: Image) -> bool:
    image_np = np.array(image)
    if np.all(image_np == 0):
        return True

    safety_feature_extractor, safety_checker = resource_management.SingletonResourceManager().get_resource(
        cst.IMAGE_SAFETY_CHECKERS
    )
    safety_checker_device = resource_management.SingletonResourceManager()._config[cst.IMAGE_SAFETY_CHECKERS]
    with autocast():
        safety_checker_input = safety_feature_extractor(images=image, return_tensors="pt").to(safety_checker_device)
        result, has_nsfw_concepts = safety_checker.forward(clip_input=safety_checker_input.pixel_values, images=image)

        return has_nsfw_concepts


def get_seed_generator(seed: int) -> torch.Generator:
    return torch.manual_seed(seed)


def resize_image(image_b64: str) -> str:
    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data))

    best_size = find_closest_allowed_size(image)
    resized_image = image.resize(best_size, Image.Resampling.BICUBIC)

    byte_arr = BytesIO()
    resized_image.save(byte_arr, format="PNG")
    encoded_resized_image = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    return encoded_resized_image


def get_closest_mutliple_of_64(number: int) -> int:
    return 64 * (math.ceil(number / 64))


def find_closest_allowed_size(image) -> Tuple[int, int]:
    width, height = image.size
    min_diff: float = float("inf")
    best_size: Tuple[int, int] = None
    for size in cst.ALLOWED_IMAGE_SIZES:
        diff = abs(width - size[0]) + abs(height - size[1])
        if diff < min_diff:
            min_diff = diff
            best_size = size
    return best_size


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
    return [rle_decode(rle, shape) for rle in rles]


def get_positive_and_negative_prompts(text_prompts: List[dc.TextPrompt]) -> Tuple[str, str]:
    positive_prompt = ""
    negative_prompt = ""

    for prompt in text_prompts:
        if prompt.weight is None or prompt.weight >= 0:
            positive_prompt += prompt.text
        else:
            negative_prompt += prompt.text

    return positive_prompt, negative_prompt


def image_hash_feature_extraction(image: Image.Image) -> utility_models.ImageHashes:

    phash = str(imagehash.phash(image))
    ahash = str(imagehash.average_hash(image))
    dhash = str(imagehash.dhash(image))
    chash = str(imagehash.colorhash(image))

    return utility_models.ImageHashes(
        perceptual_hash=phash,
        average_hash=ahash,
        difference_hash=dhash,
        color_hash=chash,
    )

def get_clip_embedding_from_processed_image(image: Image.Image) -> List[float]:

    threading_lock = resource_management.threading_lock

    clip_model, clip_processor = resource_management.SingletonResourceManager().get_resource(cst.MODEL_CLIP)
    clip_device = resource_management.SingletonResourceManager()._config.get(cst.MODEL_CLIP)

    with threading_lock:
        processed_images = [clip_processor(image)]
        processed_images_tensor = torch.stack(processed_images).to(clip_device)

        with torch.no_grad():
            image_embeddings = clip_model.encode_image(processed_images_tensor)

    image_embeddings = image_embeddings.cpu().numpy().tolist()
    return image_embeddings