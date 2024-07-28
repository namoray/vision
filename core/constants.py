from typing import List, Tuple


SCORING_PERIOD_TIME = 60 * 60  # 0.5

SEED_CHUNK_SIZE = 1_000_000


TASK_CONFIG_JSON = "task_config.json"
TASK_CONCURRENCY_CONFIG_JSON = "task_concurrency_config.json"
BLOCKS_PER_EPOCH = 360
BLOCK_TIME_IN_S = 12


SPECIAL_CONCEPTS = [
    "little girl",
    "young child",
    "young girl",
    "little boy",
    "young boy",
]
NSFW_CONCEPTS = [
    "sexual",
    "nude",
    "sex",
    "18+",
    "naked",
    "nsfw",
    "porn",
    "dick",
    "vagina",
    "naked child",
    "explicit content",
    "uncensored",
    "fuck",
    "nipples",
    "visible nipples",
    "naked breasts",
    "areola",
]
NSFW_RESPONSE_ERROR = "error_nsfw_image"

LUNA_DIFFUSION_REPO = "proximasanfinetuning/luna-diffusion"

DEFAULT_NEGATIVE_PROMPT = ", ".join(NSFW_CONCEPTS) + ", worst quality, low quality"
KANDINKSY_NEGATIVE_PROMPT_PERFIX = "overexposed"

LARGEST_SEED = 4294967295
DEFAULT_CFG_SCALE = 7
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_SAMPLES = 1
DEFAULT_STEPS = 4
DEFAULT_STYLE_PRESET = None
DEFAULT_IMAGE_STRENGTH = 0.20
DEFAULT_INIT_IMAGE_MODE = "IMAGE_STRENGTH"
DEFAULT_SAMPLER = None
DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-0"
UPSCALE_ENGINE = "esrgan-v1-x2plus"

ALLOWED_IMAGE_SIZES: List[Tuple[int, int]] = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
    (640, 1536),
    (768, 1344),
    (832, 1216),
    (896, 1152),
]

DEBUG_MINER_PARAM = "debug_miner"

MODEL_CLIP = "clip"


AVAILABLE_TASKS_OPERATION = "available_tasks_operation"

MODEL_SDXL_TURBO = "sdxl_turbo"
MODEL_SCRIBBLE = "scribble"
MODEL_MARKOV = "markov"
MODEL_CACHE = "cache"
MODEL_UPSCALE = "upscale"

PROMPT_SAFETY_CHECKERS = "prompt_safety_checkers"
IMAGE_SAFETY_CHECKERS = "image_safety_checkers"


DEVICE_DEFAULT = "cuda"


CONFIG_FILEPATH = "config.yaml"


OPERATION_TEXT_TO_IMAGE = "TextToImage"
OPERATION_IMAGE_TO_IMAGE = "ImageToImage"
OPERATION_INPAINT = "Inpaint"
OPERATION_UPSCALE = "Upscale"
OPERATION_SEGMENT = "Segment"
OPERATION_CLIP_EMBEDDINGS = "ClipEmbeddings"
OPERATION_SCRIBBLE = "Scribble"


CLIP_MODEL_REPO = "ViT-B/32"
DREAMSHAPER_XL_LOCAL_FILE = "dreamshaper_xl.safetensors"
KANDINSKY_PIPELINE_REPO = "kandinsky-community/kandinsky-2-2-decoder"
INPAINT_PIPELINE_REPO = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
DREAMSHAPER_PIPELINE_REPO = "Lykon/DreamShaper"
CONTROL_MODEL_REPO = "lllyasviel/control_v11p_sd15_scribble"
DATASET_REPO = "multi-train/coco_captions_1107"

IS_VALIDATOR = "is_validator"

CACHE_PATH = "image_cache"
CACHE_SIZE = 40 * 1024**2  # 40mb, just something small for the validator

CHECKPOINT_PATH = "sam_vit_l_0b3195.pth"
MODEL_TYPE = "vit_l"


# If you change this, please change the git ignore too
MODELS_CACHE = "models_cache"

# Kandinsky params

PRIOR_STEPS = 25
PRIOR_GUIDANCE_SCALE = 1.0

SYNTHETIC_ENDPOINT_PREFIX = "synthetic"
CHECKING_ENDPOINT_PREFIX = "checking"
OUTGOING = "Outgoing"

SINGULAR_GPU = "SINGULAR_GPU"

HOTKEY_PARAM = "HOTKEY_NAME"


IMAGE_WORKER_URL_PARAM = "IMAGE_WORKER_URL"
MIXTRAL_TEXT_WORKER_URL_PARAM = "MIXTRAL_TEXT_WORKER_URL"
LLAMA_3_TEXT_WORKER_URL_PARAM = "LLAMA_3_TEXT_WORKER_URL"


SAFETY_CHECKERS_PARAM = "SAFETY_CHECKERS_DEVICE"
CLIP_DEVICE_PARAM = "CLIP_DEVICE"
SCRIBBLE_DEVICE_PARAM = "SCRIBBLE_DEVICE"
KANDINSKY_DEVICE_PARAM = "KANDINSKY_DEVICE"
SDXL_TURBO_DEVICE_PARAM = "SDXL_TURBO_DEVICE"
UPSCALE_DEVICE_PARAM = "UPSCALE_DEVICE"

WALLET_NAME_PARAM = "WALLET_NAME"
SUBTENSOR_NETWORK_PARAM = "SUBTENSOR_NETWORK"
SUBTENSOR_CHAINENDPOINT_PARAM = "SUBTENSOR_CHAINENDPOINT"
IS_VALIDATOR_PARAM = "IS_VALIDATOR"
API_SERVER_PORT_PARAM = "API_SERVER_PORT"
EXTERNAL_SERVER_ADDRESS_PARAM = "EXTERNAL_SERVER_ADDRESS"
AXON_PORT_PARAM = "AXON_PORT"
AXON_EXTERNAL_IP_PARAM = "AXON_EXTERNAL_IP"


VISION_DB = "vision_database.db"
