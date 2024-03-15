import base64
import io
import random
from io import BytesIO
from typing import List, Tuple

import aiohttp
import cv2
import diskcache
import numpy as np
from PIL import Image, UnidentifiedImageError
from validation.validator_api_server import similarity_comparisons
import httpx
from core import constants as cst
from core import resource_management
import re
import torch
import clip
import bittensor as bt
async def fetch_image_as_bytes(url):
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(url)
            return response.content
    except Exception as e:
        bt.logging.debug(f"Error when fetching image {url}: {e}")
        return False

def validate_gojourney_url(url):
    pattern = re.compile(r'^https:\/\/img\.midjourneyapi\.xyz\/mj\/.*\.png$')
    if pattern.match(url):
        return True
    return False

async def is_sota_image_valid(image_url: str, prompt: str) -> bool:
    
    valid_gojourney_url = validate_gojourney_url(image_url)

    if not valid_gojourney_url:
        return False

    image_bytes = await fetch_image_as_bytes(image_url)

    if not image_bytes:
        bt.logging.warning(f"Error when fetching image {image_url}, can't parse the bytes into a PIL for some reason")
        return False

    clip_model, clip_processor = resource_management.SingletonResourceManager().get_resource(cst.MODEL_CLIP)
    clip_device = resource_management.SingletonResourceManager()._config.get(cst.MODEL_CLIP)
    
    try:
        original_image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        bt.logging.warning(f"Error when fetching image {image_url}, can't parse the bytes into a PIL for some reason")

    width, height = original_image.size
    image_1 = original_image.crop((0, 0, width // 2, height // 2))
    image_2 = original_image.crop((width // 2, 0, width, height // 2))
    image_3 = original_image.crop((0, height // 2, width // 2, height))
    image_4 = original_image.crop((width // 2, height // 2, width, height))

    images = [image_1, image_2, image_3, image_4]

    images = [clip_processor(image) for image in images]
    images_tensor = torch.stack(images).to(clip_device)

    with torch.no_grad():
        image_embeddings = clip_model.encode_image(images_tensor)

    image_embeddings = image_embeddings.cpu().numpy().tolist()

    split_prompt = prompt.split("--")[0]
    texts_tensor = clip.tokenize([split_prompt]).to(clip_device)

    with torch.no_grad():
        text_embeddings = clip_model.encode_text(texts_tensor)

    text_embedding = text_embeddings.cpu().numpy().tolist()[0]

    sims = []
    for image_embedding in image_embeddings:
        embedding_sim = similarity_comparisons.get_clip_embedding_similarity(image_embedding, text_embedding)
        sims.append(embedding_sim)
    
    average_sim = sum(sims) / len(sims)

    # Found that 0.21 finds valid images >99.9% of the time, but stops invalid ones > 99% of the time too!
    if average_sim > 0.21:
        return True

    return False
    

    



def get_markov_short_sentence() -> str:
    markov_model = resource_management.SingletonResourceManager().get_resource(cst.MODEL_MARKOV)
    text = None
    while text is None:
        text = markov_model.make_short_sentence(max_chars=80)
    return text

async def get_random_image_b64(cache: diskcache.Cache) -> str:
    for key in cache.iterkeys():
        image_b64: str = cache.get(key)
        return image_b64

    return await get_random_picsum_image(random.randint(500, 1500), random.randint(500, 1500))

def store_image_b64(image_b64: str, image_uuid: str, cache: diskcache.Cache) -> None:
    cache.set(image_uuid, image_b64)

    
async def get_random_picsum_image(x_dim: int, y_dim: int) -> str:
    """
    Generate a random image with the specified dimensions, by calling unsplash api.

    Args:
        x_dim (int): The width of the image.
        y_dim (int): The height of the image.

    Returns:
        str: The base64 encoded representation of the generated image.
    """
    async with aiohttp.ClientSession() as session:
        url = f"https://picsum.photos/{x_dim}/{y_dim}"
        async with session.get(url) as resp:
            data = await resp.read()

    img = Image.open(BytesIO(data))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return img_b64


def get_outline_of_image(base64_string: str) -> str:
    image_data = base64.b64decode(base64_string)
    image_pil = Image.open(io.BytesIO(image_data)).convert("L")

    img = np.array(image_pil)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(edges, kernel, iterations=1)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)
    (_, img_outline) = cv2.threshold(img_eroded, 50, 255, cv2.THRESH_BINARY_INV)

    image_outline = Image.fromarray(img_outline)
    buffered = io.BytesIO()
    image_outline.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    return img_b64


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


def generate_mask_with_circle(image_b64: str) -> np.ndarray:
    imgdata = base64.b64decode(image_b64)
    image = Image.open(BytesIO(imgdata))
    image_np = np.array(image)

    image_shape = image_np.shape[:2]

    center_x = np.random.randint(0, image_shape[1])
    center_y = np.random.randint(0, image_shape[0])
    center = (center_x, center_y)

    mask = np.zeros(image_shape, np.uint8)

    radius = random.randint(20, 100)

    cv2.circle(mask, center, radius, (1), 1)

    mask = cv2.floodFill(mask, None, center, 1)[1]
    mask_img = Image.fromarray(mask, "L")
    buffered = BytesIO()
    mask_img.save(buffered, format="PNG")
    mask_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return mask_img_str
