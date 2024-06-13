from enum import Enum
from typing import Dict


class Task(Enum):
    chat_mixtral = "chat-mixtral"
    chat_llama_3 = "chat-llama-3"
    proteus_text_to_image = "proteus-text-to-image"
    playground_text_to_image = "playground-text-to-image"
    dreamshaper_text_to_image = "dreamshaper-text-to-image"
    proteus_image_to_image = "proteus-image-to-image"
    playground_image_to_image = "playground-image-to-image"
    dreamshaper_image_to_image = "dreamshaper-image-to-image"
    jugger_inpainting = "inpaint"
    clip_image_embeddings = "clip-image-embeddings"
    avatar = "avatar"


TASK_TO_MAX_CAPACITY: Dict[Task, int] = {
    Task.chat_mixtral: 576_000,
    Task.chat_llama_3: 576_000,
    Task.proteus_text_to_image: 3_600,
    Task.playground_text_to_image: 10_000,
    Task.dreamshaper_text_to_image: 3_000,
    Task.proteus_image_to_image: 3_600,
    Task.playground_image_to_image: 10_000,
    Task.dreamshaper_image_to_image: 3_000,
    Task.jugger_inpainting: 4_000,
    Task.clip_image_embeddings: 0,  # disabled clip for now
    Task.avatar: 2_120,
}
