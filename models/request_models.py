from pydantic import BaseModel, Field, root_validator, validator
from typing import Optional, List
from models import utility_models
import random
from core import constants as cst, dataclasses as dc

ALLOWED_PARAMS_FOR_ENGINE = {
    utility_models.EngineEnum.SDXL_TURBO.value: {
        "steps": {"checker": lambda x: isinstance(x, int) and x in range(3, 13),
                  "error_message": "should be an integer between 3 and 12 (inclusive)",
                  "generator": lambda: random.choice([i for i in range(3, 13)])},
        "height": {"checker": lambda h: 512 <= h <= 1920 and h % 64 == 0,
                   "error_message": "should be in between 512 and 1920 (inclusive) and multiple of 64",
                   "generator": lambda: random.choice([i for i in range(512, 1920 + 64, 64)])},
        "width": {"checker": lambda w: 512 <= w <= 1920 and w % 64 == 0,
                  "error_message": "should be in between 512 and 1920 (inclusive) and multiple of 64",
                  "generator": lambda: random.choice([i for i in range(512, 1920 + 64, 64)])},
        "cfg_scale": {"checker": lambda c: 1 <= c <= 4,
                      "error_message": "should be between 1 and 4",
                      "generator": lambda: random.random()* 3 + 1},
        "image_strength": {"checker": lambda i: 0.0 <= i <= 0.75,
                           "error_message": "should be between 0.0 and 0.75",
                           "generator": lambda: random.random()* 0.75},
    },
    utility_models.EngineEnum.KANDINSKY_22.value: {
        "steps": {"checker": lambda x: isinstance(x, int) and x in range(20, 41),
                  "error_message": "should be an integer between 20 and 40 (inclusive)",
                  "generator": lambda: random.choice([i for i in range(20, 41)])},
        "inpaint_steps": {"checker": lambda x: isinstance(x, int) and x in range(15, 25),
                          "error_message": "should be an integer between 15 and 24 (inclusive)",
                          "generator": lambda: random.choice([i for i in range(15, 25)])},
        "height": {"checker": lambda h: 512 <= h <= 1920 and h % 64 == 0,
                   "error_message": "should be in between 512 and 1920 (inclusive) and multiple of 64",
                   "generator": lambda: random.choice([i for i in range(512, 1920 + 64, 64)])},
        "width": {"checker": lambda w: 512 <= w <= 1920 and w % 64 == 0,
                  "error_message": "should be in between 512 and 1920 (inclusive) and multiple of 64",
                  "generator": lambda: random.choice([i for i in range(512, 1920 + 64, 64)])},
        "cfg_scale": {"checker": lambda c: 2 <= c <= 8,
                      "error_message": "should be between 2 and 8",
                      "generator": lambda: random.random()* 6 + 2},
        "image_strength": {"checker": lambda i: 0.0 <= i <= 0.75,
                           "error_message": "should be between 0.0 and 0.75",
                           "generator": lambda: random.random()* 0.75},
    },
    utility_models.EngineEnum.SDXL_1_5.value: {
        "steps": {"checker": lambda x: isinstance(x, int) and x in range(7, 26),
                  "error_message": "should be an integer between 7 and 25 (inclusive)",
                  "generator": lambda: random.choice([i for i in range(7, 26)])},
        "height": {"checker": lambda h: 512 <= h <= 1920 and h % 64 == 0,
                   "error_message": "should be in between 512 and 1920 (inclusive) and multiple of 64",
                   "generator": lambda: random.choice([i for i in range(512, 1920 + 64, 64)])},
        "width": {"checker": lambda w: 512 <= w <= 1920 and w % 64 == 0,
                  "error_message": "should be in between 512 and 1920 (inclusive) and multiple of 64",
                  "generator": lambda: random.choice([i for i in range(512, 1920 + 64, 64)])},
        "cfg_scale": {"checker": lambda c: 2 <= c <= 8,
                      "error_message": "should be between 2 and 8",
                      "generator": lambda: random.random()* 6 + 2},
        "image_strength": {"checker": lambda i: 0.0 <= i <= 0.75,
                           "error_message": "should be between 0.0 and 0.75",
                           "generator": lambda: random.random()* 0.75},
    },
}

class TextToImageRequest(BaseModel):
    """Generate an image from text!

    Supports model with various allowed parameters. Models:

    - SDXL TURBO
    -- steps: 3 to 12
    -- height: 512 to 1920 (and multiple of 64)
    -- width: 512 to 1920 (and multiple of 64)
    -- cfg_scale: 1 to 4

    - Kandinsky 2.2
    -- steps: 25 to 40
    -- height: 512 to 1920 (and multiple of 64)
    -- width: 512 to 1920 (and multiple of 64)
    -- cfg_scale: 2 to 8
    """

    class Config:
        use_enum_values = True
        extra = "forbid"

    cfg_scale: float = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process - must be an ")
    engine: utility_models.EngineEnum = Field(
        default=utility_models.EngineEnum.SDXL_TURBO.value, description="The engine to use for image generation"
    )


    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")
    height: int = Field(cst.DEFAULT_HEIGHT, description="Height of the generated image")
    width: int = Field(cst.DEFAULT_WIDTH, description="Width of the generated image")

    @validator("text_prompts")
    def check_text_prompts_non_empty(cls, values):
        if not values:
            raise ValueError("Text prompts cannot be empty")
        return values
    @root_validator
    def allowed_params_validator(cls, values):

        engine = values.get('engine')
        steps = values.get('steps')
        height = values.get('height')
        width = values.get('width')
        cfg_scale = values.get('cfg_scale')

        params_and_values = [("steps", steps), ("height", height), ("width", width), ("cfg_scale", cfg_scale)]

        if engine not in ALLOWED_PARAMS_FOR_ENGINE:
            raise ValueError(f"Engine {engine} not supported")
        allowed_params = ALLOWED_PARAMS_FOR_ENGINE[engine]
        for param, value in params_and_values:
            checker = allowed_params[param]['checker']
            if not checker(value):
                error_message = allowed_params[param]['error_message']
                raise ValueError(f"Invalid value {value} provided for {param}, with engine {engine}. The value {error_message}")

        return values



class ImageToImageRequest(BaseModel):
    """Generate an image from another image (+ text)!

    Supports model with various allowed parameters. Models:

    - SDXL TURBO
    -- steps: 3 to 12
    -- height: 512 to 1920 (and multiple of 64)
    -- width: 512 to 1920 (and multiple of 64)
    -- cfg_scale: 1 to 4
    -- image_strength: 0.0 to 0.75

    - Kandinsky 2.2
    -- steps: 25 to 40
    -- height: 512 to 1920 (and multiple of 64)
    -- width: 512 to 1920 (and multiple of 64)
    -- cfg_scale: 2 to 8
    -- image_strength: 0.0 to 0.75
    """


    class Config:
        use_enum_values = True

    cfg_scale: float = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process")
    engine: utility_models.EngineEnum = Field(
        default=utility_models.EngineEnum.SDXL_TURBO.value, description="The engine to use for image generation"
    )


    init_image: str = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    image_strength: float = Field(0.25, description="The strength of the init image")

    height: Optional[int] = Field(None, description="Height of the generated image")
    width: Optional[int] = Field(None, description="Width of the generated image")

    @validator("text_prompts")
    def check_text_prompts_non_empty(cls, values):
        if not values:
            raise ValueError("Text prompts cannot be empty")
        return values
    @root_validator
    def allowed_params_validator(cls, values):

        engine = values.get('engine')
        steps = values.get('steps')
        height = values.get('height')
        width = values.get('width')
        cfg_scale = values.get('cfg_scale')
        image_strength = values.get('image_strength')



        params_and_values = [("steps", steps), ("height", height), ("width", width), ("cfg_scale", cfg_scale), ("image_strength", image_strength)]

        if engine not in ALLOWED_PARAMS_FOR_ENGINE:
            raise ValueError(f"Engine {engine} not supported")
        allowed_params = ALLOWED_PARAMS_FOR_ENGINE[engine]
        for param, value in params_and_values:
            checker = allowed_params[param]
            if not checker(value):
                raise ValueError(f"Value {value} for {param} not allowed for the engine {engine}.")

        return values

class InpaintRequest(BaseModel):

    class Config:
        use_enum_values = True

    cfg_scale: float = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process")
    engine: utility_models.EngineEnum = Field(
        default=utility_models.EngineEnum.SDXL_TURBO.value, description="The engine to use for image generation"
    )

    init_image: str = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    mask_image: str = Field(None, description="The base64 encoded mask", title="mask_source")

    @validator("text_prompts")
    def check_text_prompts_non_empty(cls, values):
        if not values:
            raise ValueError("Text prompts cannot be empty")
        return values
    @root_validator
    def allowed_params_validator(cls, values):

        engine = values.get('engine')
        steps = values.get('steps')
        cfg_scale = values.get('cfg_scale')



        params_and_values = [("inpaint_steps", steps), ("cfg_scale", cfg_scale)]

        if engine not in ALLOWED_PARAMS_FOR_ENGINE:
            raise ValueError(f"Engine {engine} not supported")
        allowed_params = ALLOWED_PARAMS_FOR_ENGINE[engine]
        for param, value in params_and_values:
            checker = allowed_params[param]
            if not checker(value):
                raise ValueError(f"Value {value} for {param} not allowed for the engine {engine}.")

        return values

class ScribbleRequest(BaseModel):
    """Generate an image from a doodle (+ text)!

    Supports model with various allowed parameters. Models:

    - SD-1.5:
    -- steps: 7 to 25
    -- height: 512 to 1920 (and multiple of 64)
    -- width: 512 to 1920 (and multiple of 64)
    -- cfg_scale: 2 to 8
    -- image_strength: 0.0 to 0.75
    """
    class Config:
        use_enum_values = True

    cfg_scale: float = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process")
    engine: utility_models.EngineEnum = Field(
        default=utility_models.EngineEnum.SDXL_1_5.value, description="The engine to use for image generation", const=True
    )

    init_image: str = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")

    engine: utility_models.EngineEnum = Field(utility_models.EngineEnum.SDXL_1_5, const=True)
    image_strength: float = Field(0.25, description="The strength of the init image")

    height: Optional[int] = Field(None, description="Height of the generated image")
    width: Optional[int] = Field(None, description="Width of the generated image")

    @validator("text_prompts")
    def check_text_prompts_non_empty(cls, values):
        if not values:
            raise ValueError("Text prompts cannot be empty")
        return values
    @root_validator
    def allowed_params_validator(cls, values):

        engine = values.get('engine')
        steps = values.get('steps')
        height = values.get('height')
        width = values.get('width')
        cfg_scale = values.get('cfg_scale')
        image_strength = values.get('image_strength')



        params_and_values = [("steps", steps), ("height", height), ("width", width), ("cfg_scale", cfg_scale), ("image_strength", image_strength)]

        if engine not in ALLOWED_PARAMS_FOR_ENGINE:
            raise ValueError(f"Engine {engine} not supported")
        allowed_params = ALLOWED_PARAMS_FOR_ENGINE[engine]
        for param, value in params_and_values:
            checker = allowed_params[param]
            if not checker(value):
                raise ValueError(f"Value {value} for {param} not allowed.")

        return values


class UpscaleRequest(BaseModel):
    image: str = Field(..., description="The base64 encoded image", title="image")


class ClipEmbeddingsRequest(BaseModel):

    image_b64s: List[str] = Field(
        None,
        description="The image b64s",
        title="image_b64s",
    )


class TextToImageResponse(BaseModel):
    image_b64s: List[str] = Field(..., description="The base64 encoded images to return", title="image_b64s")

class ImageToImageResponse(BaseModel):
    image_b64s: List[str] = Field(..., description="The base64 encoded images to return", title="image_b64s")

class InpaintResponse(BaseModel):
    image_b64s: List[str] = Field(..., description="The base64 encoded images to return", title="image_b64s")

class ScribbleResponse(BaseModel):
    image_b64s: List[str] = Field(..., description="The base64 encoded images to return", title="image_b64s")

class UpscaleResponse(BaseModel):
    image_b64s: List[str] = Field(..., description="The base64 encoded images to return", title="image_b64s")

class ClipEmbeddingsResponse(BaseModel):
    image_embeddings: List[List[float]] = Field(..., description="The image embeddings", title="image_embeddings")
