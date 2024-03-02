
from models import base_models, utility_models
import xgboost as xgb
from typing import List
import imagehash
import numpy as np


images_are_same_classifier = xgb.XGBClassifier()
images_are_same_classifier.load_model('image_similarity_xgb_model.json')

def _hash_distance(hash_1: str, hash_2: str) -> int:

    restored_hash1 = imagehash.hex_to_hash(hash_1)
    restored_hash2 = imagehash.hex_to_hash(hash_2)

    return restored_hash1 - restored_hash2


def _get_hash_distances(hashes_1: utility_models.ImageHashes, hashes_2: utility_models.ImageHashes) -> List[int, int, int, int]:

    ahash_distance = _hash_distance(hashes_1.average_hash, hashes_2.average_hash)
    phash_distance = _hash_distance(hashes_1.perceptual_hash, hashes_2.perceptual_hash)
    dhash_distance = _hash_distance(hashes_1.difference_hash, hashes_2.difference_hash)
    chash_distance = _hash_distance(hashes_1.color_hash, hashes_2.color_hash)

    return [phash_distance, ahash_distance, dhash_distance, chash_distance]

def _images_are_different_probability(formatted_response1: base_models.ImageResponseBase , formatted_response2: base_models.ImageResponseBase) -> float:

    # If one is None, then return 0 if they are both None, else 1
    if formatted_response1 is None or formatted_response2 is None:
        return float(formatted_response1 != formatted_response2)

    # Else if the images returned are empty or None, then return 0 of they are both the same, else 1
    elif formatted_response1.image_b64s is None or formatted_response2.image_b64s is None or len(formatted_response1.image_b64s) == 0 or len(formatted_response2.image_b64s) == 0:
        return float(formatted_response1.image_b64s != formatted_response2.image_b64s)

    model_features = _get_hash_distances(formatted_response1.image_hashes[0], formatted_response2.image_hashes[0])

    probability_different_image = images_are_same_classifier.predict_proba([model_features])[0][0]

    return probability_different_image


def images_are_same_generic(formatted_response1: base_models.ImageResponseBase , formatted_response2: base_models.ImageResponseBase) -> float:
    return _images_are_different_probability(formatted_response1, formatted_response2) < 0.99

def images_are_same_upscale(formatted_response1: base_models.ImageResponseBase , formatted_response2: base_models.ImageResponseBase) -> float:
    return _images_are_different_probability(formatted_response1, formatted_response2) < 0.9

def clip_embeddings_are_same(formatted_response1: base_models.ClipEmbeddingsBase , formatted_response2: base_models.ClipEmbeddingsBase) -> float:

    if formatted_response1 is None or formatted_response2 is None:
        return float(formatted_response1 != formatted_response2)
    elif formatted_response1.image_embeddings is None or formatted_response2.image_embeddings is None:
        return float(formatted_response1.image_embeddings != formatted_response2.image_embeddings)
    else:
        image_embedding1 = np.array(formatted_response1.image_embeddings, dtype=float)
        image_embedding2 = np.array(formatted_response2.image_embeddings, dtype=float)

        dot_product = np.dot(image_embedding1, image_embedding2.T)
        norm1 = np.linalg.norm(image_embedding1)
        norm2 = np.linalg.norm(image_embedding2)

        normalized_dot_product = dot_product / (norm1 * norm2)

        return float(normalized_dot_product[0][0] > 0.995)






# ADD ONE FOR CLIP EMBEDDINGS

SYNAPSE_TO_COMPARISON_FUNCTION = {
    "TextToImage": images_are_same_generic,
    "ImageToImage": images_are_same_generic,
    "Inpaint": images_are_same_generic,
    "Scribble": images_are_same_generic,
    "Upscale": images_are_same_upscale,
    "ClipEmbeddings": clip_embeddings_are_same,
}
