import base64
import os

import cv2
import numpy as np
import torch
from torch import nn
from typing import List, Optional, Dict
from PIL import Image
import io
import binascii
from core import constants as cst
from RealESRGAN import RealESRGAN
import yaml
from pydantic import BaseModel


from RealESRGAN.utils import pad_reflect, split_image_into_overlapping_patches, stich_together, unpad_image

import gc


class RealESRGANClass(RealESRGAN):
    def to(self, device: str) -> None:
        self.device = device
        self.model.to(self.device)
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.cuda.amp.autocast()
    def predict(self, lr_image, batch_size=4, patches_size=192, padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(lr_image, patch_size=patches_size, padding_size=padding)
        img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()

        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i : i + batch_size])), 0)

        sr_image = res.permute((0, 2, 3, 1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding * scale,
        )
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * scale)
        sr_img = Image.fromarray(sr_img)

        del lr_image, img, res, np_sr_image, sr_image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return sr_img


def pascal_to_snake(pascal_str: str):
    snake_str = "".join(["_" + i.lower() if i.isupper() else i for i in pascal_str]).lstrip("_")
    return snake_str


def get_similarity_score_from_image_b64s(
    expected_b64s: Optional[List[str]], response_b64s: Optional[List[str]]
) -> float:
    """
    Calculates the similarity of two images given their base64 representation.

    - Dot product to find the similarity to give a value -1 <= x <= 1
    - If < 0 then it's pure garbage
    - Else then divide the ratios of the smallest to largest to penalise magnitude deviations, and multiply that by the sim

    The way to maximise this function is when the expected images (at each index) are identical. Any deviation from that will result in
    a lower score
    """
    similarites = []
    if expected_b64s is None or response_b64s is None:
        return float(expected_b64s == response_b64s)

    for b64_img1, b64_img2 in zip(expected_b64s, response_b64s):
        try:
            byte_img1 = base64.b64decode(b64_img1)
            byte_img2 = base64.b64decode(b64_img2)
        except binascii.Error:
            return 0

        np_img1 = np.array(Image.open(io.BytesIO(byte_img1)))
        np_img2 = np.array(Image.open(io.BytesIO(byte_img2)))

        flattened_img1 = np_img1.flatten().astype(float)
        flattened_img2 = np_img2.flatten().astype(float)

        norm1 = np.linalg.norm(flattened_img1)
        norm2 = np.linalg.norm(flattened_img2)

        if norm1 == 0 or norm2 == 0:
            similarites.append(float(np.all(flattened_img1 == 0) and np.all(flattened_img2 == 0)))
            continue

        cosine_sim = np.dot(flattened_img1, flattened_img2) / (norm1 * norm2)
        if cosine_sim <= 0:
            similarites.append(0)
            continue
        sim = cosine_sim * (min(norm1, norm2) / max(norm1, norm2))
        sim = sim**2
        similarites.append(round(sim, 3))

    return sum(similarites) / len(similarites) if len(similarites) > 0 else 0


def set_stuff_for_deterministic_output():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.benchmark = False

    # Below is disabled as it causes major issues for loading and unloading models
    # And it doesn't help much with determinsm anyway with seed
    # The Clip fallback majorly solves this problem
    # Also, the reward model is also designed to compare images on different hardware
    # and predict if they were the same or not

    # torch.use_deterministic_algorithms(True)


def get_b64_from_pipeline_image(processed_image: torch.Tensor) -> str:
    torch.cuda.empty_cache()
    np_image = np.array(processed_image)[:, :, ::-1]
    np_image = cv2.imencode(".png", np_image)[1].tobytes()
    b64_image = base64.b64encode(np_image).decode("utf-8")
    return b64_image


def pil_to_base64(image: Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_to_pil(image_b64: str) -> Image.Image:
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        return image
    except binascii.Error:
        return None


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


@torch.no_grad()
def forward_inspect(self, clip_input, images):
    pooled_output = self.vision_model(clip_input)[1]
    image_embeds = self.visual_projection(pooled_output)

    special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
    cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

    matches = {"nsfw": [], "special": []}
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {
            "special_scores": {},
            "special_care": [],
            "concept_scores": {},
            "bad_concepts": [],
        }

        adjustment = 0.0

        for concet_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concet_idx]
            concept_threshold = self.special_care_embeds_weights[concet_idx].item()
            result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
            if result_img["special_scores"][concet_idx] > 0:
                result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
                adjustment = 0.01
                matches["special"].append(cst.SPECIAL_CONCEPTS[concet_idx])

        for concet_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concet_idx]
            concept_threshold = self.concept_embeds_weights[concet_idx].item()
            result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)

            if result_img["concept_scores"][concet_idx] > 0:
                result_img["bad_concepts"].append(concet_idx)
                matches["nsfw"].append(cst.NSFW_CONCEPTS[concet_idx])

    has_nsfw_concepts = len(matches["nsfw"]) > 0

    return matches, has_nsfw_concepts


def get_validator_hotkey_name_from_config(yaml_config: Dict[str, str]) -> str:
    yaml_config = yaml.safe_load(open(cst.CONFIG_FILEPATH))
    for hotkey, config in yaml_config.items():
        if config["IS_VALIDATOR"]:
            return hotkey

    raise ValueError("Please set up the config for a validator!")


# set_stuff_for_deterministic_output()


def model_to_printable_dict(model: Optional[BaseModel], max_length: int = 50) -> dict:
    """
    Convert a model to a dictionary, truncating long string values and string representation of lists.
    Helper function to print synapses & stuff with image b64's in them

    Parameters:
    max_length (int): The maximum length allowed for string fields. Default is 30.

    Returns:
    dict: The model as a dictionary with truncated values.
    """

    if model is None:
        return None

    def truncate_value(value):
        if isinstance(value, str) and len(value) > max_length:
            return value[:max_length] + "..."
        elif isinstance(value, list):
            str_value = str(value)
            if len(str_value) >= max_length:
                return str(value)[:max_length] + "..."
            else:
                return str_value
        elif isinstance(value, dict):
            return {k: truncate_value(v) for k, v in value.items()}
        else:
            return value

    try:
        model_dict = model.dict()
        printable_dict = {k: truncate_value(v) for k, v in model_dict.items()}
        return printable_dict
    except TypeError:
        return {}


set_stuff_for_deterministic_output()
