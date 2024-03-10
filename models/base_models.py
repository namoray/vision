"""
The naming convention is super important to adhere too!

Keep it as SynapseNameBase / SynapseNameIncoming / SynapseNameOutgoing
"""
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from core import constants as cst
from core import dataclasses as dc
from models import utility_models
import bittensor as bt

class BaseSynapse(bt.Synapse):
    error_message: Optional[str] = Field(None)

class BaseOutgoing(BaseModel):
    error_message: Optional[str] = Field(None, description="The error message", title="error_message")


# AVAILABLE OPERATIONS


class AvailableOperationsIncoming(BaseModel):
    ...


class AvailableOperationsOutgoing(BaseModel):
    available_operations: Optional[Dict[str, bool]]


class AvailableOperationsBase(AvailableOperationsIncoming, AvailableOperationsOutgoing):
    ...


# Generic image gen


class ImageGenerationBase(BaseModel):
    cfg_scale: float = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process")
    seed: int = Field(..., description="Random seed for generating the image. NOTE: THIS CANNOT BE SET, YOU MUST PASS IN 0, SORRY!")
    engine: utility_models.EngineEnum = Field(
        default=utility_models.EngineEnum.SDXL_TURBO.value, description="The engine to use for image generation"
    )

    class Config:
        use_enum_values = True

class ImageResponseBase(BaseOutgoing):
    image_b64s: Optional[List[str]] = Field(None, description="The base64 encoded images to return", title="image_b64s")
    clip_embeddings: Optional[List[List[float]]] = Field(None, description="The clip embeddings for each of the images")
    image_hashes: Optional[List[utility_models.ImageHashes]] = Field(None, description="Image hash's for each image")


# TEXT TO IMAGE
class TextToImageIncoming(ImageGenerationBase):
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    height: int = Field(cst.DEFAULT_HEIGHT, description="Height of the generated image")
    width: int = Field(cst.DEFAULT_WIDTH, description="Width of the generated image")


class TextToImageOutgoing(ImageResponseBase):
    ...


class TextToImageBase(TextToImageIncoming, TextToImageOutgoing):
    ...


# IMAGE TO IMAGE


class ImageToImageIncoming(ImageGenerationBase):
    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    init_image_mode: Optional[str] = Field("IMAGE_STRENGTH", description="The mode of the init image")
    image_strength: float = Field(0.25, description="The strength of the init image")

    height: Optional[int] = Field(None, description="Height of the generated image")
    width: Optional[int] = Field(None, description="Width of the generated image")


class ImageToImageOutgoing(ImageResponseBase):
    ...

class ImageToImageBase(ImageToImageIncoming, ImageToImageOutgoing):
    ...


# Inpaint


class InpaintIncoming(ImageGenerationBase):
    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    mask_image: Optional[str] = Field(None, description="The base64 encoded mask", title="mask_source")

    class Config:
        use_enum_values = True


class InpaintOutgoing(ImageResponseBase):
    ...


class InpaintBase(InpaintIncoming, InpaintOutgoing):
    ...


class ScribbleIncoming(ImageGenerationBase):
    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    # Overriding defaults
    engine: utility_models.EngineEnum = Field(utility_models.EngineEnum.SDXL_1_5, const=True)

    image_strength: float = Field(0.25, description="The strength of the init image")

    height: Optional[int] = Field(None, description="Height of the generated image")
    width: Optional[int] = Field(None, description="Width of the generated image")


class ScribbleOutgoing(ImageResponseBase):
    ...


class ScribbleBase(ScribbleIncoming, ScribbleOutgoing):
    ...


# Upscale


class UpscaleIncoming(BaseModel):
    image: Optional[str] = Field(..., description="The base64 encoded image", title="image")


class UpscaleOutgoing(ImageResponseBase):
    ...


class UpscaleBase(UpscaleIncoming, UpscaleOutgoing):
    ...


# CLIP EMBEDDINGS
class ClipEmbeddingsIncoming(BaseModel):

    image_b64s: Optional[List[str]] = Field(
        None,
        description="The image b64s",
        title="image_b64s",
    )


class ClipEmbeddingsOutgoing(BaseOutgoing):
    image_embeddings: Optional[List[List[float]]] = Field(None, description="The image embeddings", title="image_embeddings")


class ClipEmbeddingsBase(ClipEmbeddingsIncoming, ClipEmbeddingsOutgoing):
    ...

# SOTA
class SOTAIncoming(BaseModel):
    prompt: str

class SOTAOutgoing(BaseModel):
    image_url: str

# SAM
class SegmentIncoming(BaseModel):

    # TODO: Re-enable in phase 2
    # image_uuid: Optional[str] = Field(None, description="The UUID for the image to be segmented", title="embedding")

    image_b64: str = Field(None, description="The base64 encoded image", title="image")

    input_points: Optional[List[List[Union[float, int]]]] = Field(
        default=None,
        description="The json encoded points for the image",
        title="points",
    )
    input_labels: Optional[List[int]] = Field(
        default=None,
        description="The labels of the points. 1 for a positive point, 0 for a negative",
        title="labels",
    )
    input_boxes: Optional[Union[List[List[Union[int, float]]], List[Union[int, float]]]] = Field(
        default=None,
        description="The boxes for the image. For now, we only accept one",
        title="boxes",
    )


class SegmentOutgoing(BaseOutgoing):
    image_uuid: Optional[str] = Field(None, description="The UUID for the image to be segmented", title="embedding")

    image_shape: Optional[List[int]] = Field(
        default=None,
        description="The shape of the image (y_dim, x_dim)",
        title="image_shape",
    )
    masks: Optional[List[List[List[int]]]] = Field(
        default=None,
        description="The json encoded RLE masks",
        title="masks",
    )


class SegmentBase(SegmentIncoming, SegmentOutgoing):
    ...
