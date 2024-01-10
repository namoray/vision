import base64
import io

import bittensor as bt
import torch
from PIL import Image

from core import constants as cst
from core import resource_management
from models import base_models


async def clip_embeddings_logic(
    body: base_models.ClipEmbeddingsIncoming,
) -> base_models.ClipEmbeddingsOutgoing:
    threading_lock = resource_management.threading_lock

    clip_model, clip_processor = resource_management.SingletonResourceManager().get_resource(cst.MODEL_CLIP)
    clip_device = resource_management.SingletonResourceManager()._config.get(cst.MODEL_CLIP)

    output = base_models.ClipEmbeddingsOutgoing()

    if body.image_b64s is None:
        output.error_message = "❌ You must supply the images that you want to embed"
        bt.logging.warning(f"FAULT OF THE API USER: {output.error_message}, body: {body}")
        return output

    images = [Image.open(io.BytesIO(base64.b64decode(img_b64))) for img_b64 in body.image_b64s]
    with threading_lock:
        images = [clip_processor(image) for image in images]
        images_tensor = torch.stack(images).to(clip_device)

        with torch.no_grad():
            image_embeddings = clip_model.encode_image(images_tensor)

    image_embeddings = image_embeddings.cpu().numpy().tolist()
    output.image_embeddings = image_embeddings

    if len(image_embeddings) > 0:
        bt.logging.info(f"✅ {len(output.image_embeddings)} image embedding(s) generated. bang.")

    return output
