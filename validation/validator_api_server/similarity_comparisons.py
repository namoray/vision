
from models import base_models
from core import utils as core_utils
import xgboost as xgb
from typing import Tuple
import imagehash
import numpy as np
images_are_same_classifier = xgb.XGBClassifier()
images_are_same_classifier.load_model('image_similarity_xgb_model.json')

def _images_are_different_probability(formatted_response1: base_models.ImageResponseBase , formatted_response2: base_models.ImageResponseBase) -> float:

    if formatted_response1 is None or formatted_response2 is None:
        return float(formatted_response1 != formatted_response2)
    elif formatted_response1.image_b64s is None or formatted_response2.image_b64s is None or len(formatted_response1.image_b64s) == 0 or len(formatted_response2.image_b64s) == 0:
        return float(formatted_response1.image_b64s != formatted_response2.image_b64s)

    model_features = _image_hash_feature_extraction(formatted_response1.image_b64s[0], formatted_response2.image_b64s[0])

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

def _image_hash_feature_extraction(image1_b64: str, image2_b64: str) -> Tuple[float, float]:

    image1 = core_utils.base64_to_pil(image1_b64)
    image2 = core_utils.base64_to_pil(image2_b64)

    phash1 = imagehash.phash(image1)
    phash2 = imagehash.phash(image2)
    phash_distance = phash1 - phash2

    ahash1 = imagehash.average_hash(image1)
    ahash2 = imagehash.average_hash(image2)
    ahash_distance = ahash1 - ahash2

    dhash1 = imagehash.dhash(image1)
    dhash2 = imagehash.dhash(image2)
    dhash_distance = dhash1 - dhash2

    chash1 = imagehash.colorhash(image1)
    chash2 = imagehash.colorhash(image2)
    chash_distance = chash1 - chash2

    return [phash_distance, ahash_distance, dhash_distance, chash_distance]




# ADD ONE FOR CLIP EMBEDDINGS

SYNAPSE_TO_COMPARISON_FUNCTION = {
    "TextToImage": images_are_same_generic,
    "ImageToImage": images_are_same_generic,
    "Inpaint": images_are_same_generic,
    "Scribble": images_are_same_generic,
    "Upscale": images_are_same_upscale,
    "ClipEmbeddings": clip_embeddings_are_same,
}
