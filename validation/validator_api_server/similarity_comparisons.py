from models import base_models, utility_models
import xgboost as xgb
from typing import List, Tuple
import imagehash
import numpy as np
import bittensor as bt

images_are_same_classifier = xgb.XGBClassifier()
images_are_same_classifier.load_model("image_similarity_xgb_model.json")


def _hash_distance(hash_1: str, hash_2: str) -> int:
    restored_hash1 = imagehash.hex_to_hash(hash_1)
    restored_hash2 = imagehash.hex_to_hash(hash_2)

    return restored_hash1 - restored_hash2


def _get_hash_distances(hashes_1: utility_models.ImageHashes, hashes_2: utility_models.ImageHashes) -> List[int]:
    ahash_distance = _hash_distance(hashes_1.average_hash, hashes_2.average_hash)
    phash_distance = _hash_distance(hashes_1.perceptual_hash, hashes_2.perceptual_hash)
    dhash_distance = _hash_distance(hashes_1.difference_hash, hashes_2.difference_hash)
    chash_distance = _hash_distance(hashes_1.color_hash, hashes_2.color_hash)

    return [phash_distance, ahash_distance, dhash_distance, chash_distance]


def _image_similarities(
    formatted_response1: base_models.ImageResponseBase, formatted_response2: base_models.ImageResponseBase
) -> Tuple[float, float]:
    # If one is None, then return 0 if they are both None, else 1
    if formatted_response1 is None or formatted_response2 is None:
        return float(formatted_response1 == formatted_response2)

    # Else if the stuff return is empty or None, then return 0 if nobody return an image, else 1
    elif (
        formatted_response1.image_b64s is None
        or formatted_response2.image_b64s is None
        or len(formatted_response1.image_b64s) == 0
        or len(formatted_response2.image_b64s) == 0
        or formatted_response1.clip_embeddings is None
        or formatted_response2.clip_embeddings is None
        or len(formatted_response1.clip_embeddings) == 0
        or len(formatted_response2.clip_embeddings) == 0
        or formatted_response1.image_hashes is None
        or formatted_response2.image_hashes is None
        or len(formatted_response1.image_hashes) == 0
        or len(formatted_response2.image_hashes) == 0
    ):
        bt.logging.info("Found empty image_b64s and what not!")
        return float(len(formatted_response1.image_b64s) == 0 and len(formatted_response2.image_b64s) == 0)

    model_features = _get_hash_distances(
        formatted_response1.image_hashes[0], formatted_response2.image_hashes[0]
    )

    probability_same_image_xg = images_are_same_classifier.predict_proba([model_features])[0][1]

    clip_similarity = get_clip_embedding_similarity(
        formatted_response1.clip_embeddings[0], formatted_response2.clip_embeddings[0]
    )

    # If they're the same by xg (it has a low threshold), then return 1, else use the clip similarity squared
    return probability_same_image_xg, clip_similarity


def images_are_same_generic(
    formatted_response1: base_models.ImageResponseBase, formatted_response2: base_models.ImageResponseBase
) -> float:
    probability_same_image_xg, clip_similarity  =  _image_similarities(formatted_response1, formatted_response2)
    return 1 if probability_same_image_xg > 0.01 else clip_similarity ** 2


def images_are_same_upscale(
    formatted_response1: base_models.ImageResponseBase, formatted_response2: base_models.ImageResponseBase
) -> float:
    probability_same_image_xg, clip_similarity  =  _image_similarities(formatted_response1, formatted_response2)
    return 1 if probability_same_image_xg > 0.08 else clip_similarity ** 4

def get_clip_embedding_similarity(
    clip_embedding1: List[float], clip_embedding2: List[float]
):
    image_embedding1 = np.array(clip_embedding1, dtype=float)
    image_embedding2 = np.array(clip_embedding2, dtype=float)

    dot_product = np.dot(image_embedding1, image_embedding2.T)
    norm1 = np.linalg.norm(image_embedding1)
    norm2 = np.linalg.norm(image_embedding2)

    normalized_dot_product = dot_product / (norm1 * norm2)

    return float(normalized_dot_product[0][0])

def clip_embeddings_are_same(
    formatted_response1: base_models.ClipEmbeddingsBase, formatted_response2: base_models.ClipEmbeddingsBase
) -> float:
    if formatted_response1 is None or formatted_response2 is None:
        return float(formatted_response1 == formatted_response2)
    elif formatted_response1.image_embeddings is None or formatted_response2.image_embeddings is None:
        return float(formatted_response1.image_embeddings == formatted_response2.image_embeddings)
    else:
        similarity = get_clip_embedding_similarity(
            formatted_response1.image_embeddings, formatted_response2.image_embeddings
        )

        return float(similarity > 0.995)


# ADD ONE FOR CLIP EMBEDDINGS

SYNAPSE_TO_COMPARISON_FUNCTION = {
    "TextToImage": images_are_same_generic,
    "ImageToImage": images_are_same_generic,
    "Inpaint": images_are_same_generic,
    "Scribble": images_are_same_generic,
    "Upscale": images_are_same_upscale,
    "ClipEmbeddings": clip_embeddings_are_same,
}
