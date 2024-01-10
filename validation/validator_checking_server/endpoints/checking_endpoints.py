from fastapi import APIRouter
from validation.validator_checking_server import utils as checking_utils
from operation_logic import utils as operation_utils
from core import resource_management, constants as core_cst
from models import base_models
from operation_logic import (
    text_to_image_logic,
    image_to_image_logic,
    inpaint_logic,
    clip_embeddings_logic,
    upscale_logic,
    scribble_logic
)

router = APIRouter()


@router.post(f"/{core_cst.CHECKING_ENDPOINT_PREFIX}/text-to-image")
async def text_to_image(request: base_models.TextToImageIncoming) -> base_models.TextToImageOutgoing:

    expected_output = await text_to_image_logic.text_to_image_logic(request)
    resource_management.SingletonResourceManager().move_all_models_to_cpu()

    if expected_output.image_b64s is not None:
        image = expected_output.image_b64s[0]
        image_uuid = operation_utils.get_image_uuid(image)
        cache = resource_management.SingletonResourceManager().get_resource(core_cst.MODEL_CACHE)
        checking_utils.store_image_b64(image, image_uuid, cache)

    return expected_output


@router.post(f"/{core_cst.CHECKING_ENDPOINT_PREFIX}/image-to-image")
async def image_to_image(request: base_models.ImageToImageIncoming) -> base_models.ImageToImageOutgoing:
    expected_output = await image_to_image_logic.image_to_image_logic(request)
    resource_management.SingletonResourceManager().move_all_models_to_cpu()
    return expected_output

@router.post(f"/{core_cst.CHECKING_ENDPOINT_PREFIX}/inpaint")
async def inpaint(request: base_models.InpaintIncoming) -> base_models.InpaintOutgoing:
    expected_output = await inpaint_logic.inpaint_logic(request)
    resource_management.SingletonResourceManager().move_all_models_to_cpu()
    return expected_output

@router.post(f"/{core_cst.CHECKING_ENDPOINT_PREFIX}/scribble")
async def scribble(request: base_models.ScribbleIncoming) -> base_models.ScribbleOutgoing:
    expected_output = await scribble_logic.scribble_logic(request)
    resource_management.SingletonResourceManager().move_all_models_to_cpu()
    return expected_output

@router.post(f"/{core_cst.CHECKING_ENDPOINT_PREFIX}/upscale")
async def upscale(request: base_models.UpscaleIncoming) -> base_models.UpscaleOutgoing:
    expected_output = await upscale_logic.upscale_logic(request)
    resource_management.SingletonResourceManager().move_all_models_to_cpu()
    return expected_output

@router.post(f"/{core_cst.CHECKING_ENDPOINT_PREFIX}/clip-embeddings")
async def clip_embeddings(request: base_models.ClipEmbeddingsIncoming) -> base_models.ClipEmbeddingsOutgoing:
    expected_output = await clip_embeddings_logic.clip_embeddings_logic(request)
    resource_management.SingletonResourceManager().move_all_models_to_cpu()
    return expected_output

# @router.post(f"/{core_cst.CHECKING_ENDPOINT_PREFIX}/segment")
# async def segment(request: base_models.SegmentIncoming) -> base_models.SegmentOutgoing:
#     expected_output = await segment_logic.segment_logic(request)
#     resource_management.SingletonResourceManager().move_all_models_to_cpu()
#     return expected_output
