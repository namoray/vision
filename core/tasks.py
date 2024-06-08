"""Would prefer to make this just one dataclass"""

from core import Task
from models import synapses, utility_models
from mining.proxy import operations
from typing import Any, Dict, Optional
import bittensor as bt

# I don't love this being here. How else should I do it though?
# I don't want to rely on any extra third party service for fetching this info...
TASK_TO_MAX_CAPACITY: Dict[Task, int] = {
    Task.chat_mixtral: 576_000,
    Task.chat_llama_3: 576_000,
    Task.proteus_text_to_image: 2_000,
    Task.playground_text_to_image: 10_000,
    Task.dreamshaper_text_to_image: 2_000,
    Task.proteus_image_to_image: 2_000,
    Task.playground_image_to_image: 6_000,
    Task.dreamshaper_image_to_image: 2_000,
    Task.jugger_inpainting: 2_000,
    Task.clip_image_embeddings: 0,  # disabled clip for now
    Task.avatar: 5_120,
}

TASK_IS_STREAM: Dict[Task, bool] = {
    Task.chat_mixtral: True,
    Task.chat_llama_3: True,
    Task.proteus_text_to_image: False,
    Task.playground_text_to_image: False,
    Task.dreamshaper_text_to_image: False,
    Task.proteus_image_to_image: False,
    Task.playground_image_to_image: False,
    Task.dreamshaper_image_to_image: False,
    Task.jugger_inpainting: False,
    Task.clip_image_embeddings: False,
    Task.avatar: False,
}
TASKS_TO_SYNAPSE: Dict[Task, bt.Synapse] = {
    Task.chat_mixtral: synapses.Chat,
    Task.chat_llama_3: synapses.Chat,
    Task.proteus_text_to_image: synapses.TextToImage,
    Task.playground_text_to_image: synapses.TextToImage,
    Task.dreamshaper_text_to_image: synapses.TextToImage,
    Task.proteus_image_to_image: synapses.ImageToImage,
    Task.playground_image_to_image: synapses.ImageToImage,
    Task.dreamshaper_image_to_image: synapses.ImageToImage,
    Task.jugger_inpainting: synapses.Inpaint,
    Task.clip_image_embeddings: synapses.ClipEmbeddings,
    Task.avatar: synapses.Avatar,
}

TASKS_TO_MINER_OPERATION_MODULES: Dict[Task, Any] = {
    Task.chat_mixtral: operations.chat_operation,
    Task.chat_llama_3: operations.chat_operation,
    Task.proteus_text_to_image: operations.text_to_image_operation,
    Task.playground_text_to_image: operations.text_to_image_operation,
    Task.dreamshaper_text_to_image: operations.text_to_image_operation,
    Task.proteus_image_to_image: operations.image_to_image_operation,
    Task.playground_image_to_image: operations.image_to_image_operation,
    Task.dreamshaper_image_to_image: operations.image_to_image_operation,
    Task.jugger_inpainting: operations.inpaint_operation,
    Task.clip_image_embeddings: operations.clip_embeddings_operation,
    Task.avatar: operations.avatar_operation,
}


def get_task_from_synapse(synapse: bt.Synapse) -> Optional[Task]:
    if isinstance(synapse, synapses.Chat):
        if synapse.model == utility_models.ChatModels.mixtral.value:
            return Task.chat_mixtral
        elif synapse.model == utility_models.ChatModels.llama_3.value:
            return Task.chat_llama_3
        else:
            return None
    elif isinstance(synapse, synapses.TextToImage):
        if synapse.engine == utility_models.EngineEnum.PROTEUS.value:
            return Task.proteus_text_to_image
        elif synapse.engine == utility_models.EngineEnum.PLAYGROUND.value:
            return Task.playground_text_to_image
        elif synapse.engine == utility_models.EngineEnum.DREAMSHAPER.value:
            return Task.dreamshaper_text_to_image
        else:
            return None
    elif isinstance(synapse, synapses.ImageToImage):
        if synapse.engine == utility_models.EngineEnum.PROTEUS.value:
            return Task.proteus_image_to_image
        elif synapse.engine == utility_models.EngineEnum.PLAYGROUND.value:
            return Task.playground_image_to_image
        elif synapse.engine == utility_models.EngineEnum.DREAMSHAPER.value:
            return Task.dreamshaper_image_to_image
        else:
            return None
    elif isinstance(synapse, synapses.Inpaint):
        return Task.jugger_inpainting
    elif isinstance(synapse, synapses.ClipEmbeddings):
        return Task.clip_image_embeddings
    elif isinstance(synapse, synapses.Avatar):
        return Task.avatar
    else:
        return None
