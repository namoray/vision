from enum import Enum
from models import synapses
from mining.proxy import operations
from config.miner_config import config as miner_config


class Tasks(Enum):
    chat_bittensor_finetune = "chat-bittensor-finetune"
    chat_mixtral = "chat-mixtral"
    proteus_text_to_image = "proteus-text-to-image"
    playground_text_to_image = "playground-text-to-image"
    dreamshaper_text_to_image = "dreamshaper-text-to-image"
    proteus_image_to_image = "proteus-image-to-image"
    playground_image_to_image = "playground-image-to-image"
    dreamshaper_image_to_image = "dreamshaper-image-to-image"
    jugger_inpainting = "inpaint"
    clip_image_embeddings = "clip-image-embeddings"
    sota = "sota"


# IF YOU ARE MINER, YOU WILL PROBABLY NEED TO FIDDLE WITH THIS:
SUPPORTED_TASKS = []
if miner_config.image_worker_url is not None:
    SUPPORTED_TASKS.extend(
        [
            Tasks.proteus_text_to_image.value,
            Tasks.playground_text_to_image.value,
            Tasks.dreamshaper_text_to_image.value,
            Tasks.proteus_image_to_image.value,
            Tasks.playground_image_to_image.value,
            Tasks.dreamshaper_image_to_image.value,
            Tasks.jugger_inpainting.value,
            Tasks.clip_image_embeddings.value,
        ]
    )
if miner_config.finetune_text_worker_url is not None:
    SUPPORTED_TASKS.extend(
        [
            Tasks.chat_bittensor_finetune.value,
        ]
    )
if miner_config.mixtral_text_worker_url is not None:
    SUPPORTED_TASKS.extend(
        [
            Tasks.chat_mixtral.value,
        ]
    )
if miner_config.sota_provider_api_key is not None:
    SUPPORTED_TASKS.extend(
        [
            Tasks.sota.value,
        ]
    )

# TODO: Do we need this?
TASKS_TO_SYNAPSE = {
    Tasks.chat_bittensor_finetune.value: synapses.Chat,
    Tasks.chat_mixtral.value: synapses.Chat,
    Tasks.proteus_text_to_image.value: synapses.TextToImage,
    Tasks.playground_text_to_image.value: synapses.TextToImage,
    Tasks.dreamshaper_text_to_image.value: synapses.TextToImage,
    Tasks.proteus_image_to_image.value: synapses.ImageToImage,
    Tasks.playground_image_to_image.value: synapses.ImageToImage,
    Tasks.dreamshaper_image_to_image.value: synapses.ImageToImage,
    Tasks.jugger_inpainting.value: synapses.Inpaint,
    Tasks.clip_image_embeddings.value: synapses.ClipEmbeddings,
    Tasks.sota.value: synapses.Sota,
}

TASKS_TO_MINER_OPERATION_MODULES = {
    Tasks.chat_bittensor_finetune.value: operations.chat_operation,
    Tasks.chat_mixtral.value: operations.chat_operation,
    Tasks.proteus_text_to_image.value: operations.text_to_image_operation,
    Tasks.playground_text_to_image.value: operations.text_to_image_operation,
    Tasks.dreamshaper_text_to_image.value: operations.text_to_image_operation,
    Tasks.proteus_image_to_image.value: operations.image_to_image_operation,
    Tasks.playground_image_to_image.value: operations.image_to_image_operation,
    Tasks.dreamshaper_image_to_image.value: operations.image_to_image_operation,
    Tasks.jugger_inpainting.value: operations.inpaint_operation,
    Tasks.clip_image_embeddings.value: operations.clip_embeddings_operation,
    Tasks.sota.value: operations.sota_operation,
}
