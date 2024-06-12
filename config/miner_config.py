from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any, Dict, Optional
from core import Task, constants as core_cst
import os
import bittensor as bt
import argparse

from mining.proxy import operations

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

def _get_env_file_from_cli_config() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", type=str, default=None)
    args, _ = parser.parse_known_args()
    env_file = args.env_file

    if not env_file:
        parser.error("You didn't specify an env file! Use --env_file to specify it.")

    return env_file


env_file = _get_env_file_from_cli_config()
if not os.path.exists(env_file):
    bt.logging.error(f"Could not find env file: {env_file}")
load_dotenv(env_file, verbose=True)


class Config(BaseModel):
    hotkey_name: str = os.getenv(core_cst.HOTKEY_PARAM, "default")
    wallet_name: str = os.getenv(core_cst.WALLET_NAME_PARAM, "default")

    subtensor_network: str = os.getenv(core_cst.SUBTENSOR_NETWORK_PARAM, "finney")
    subtensor_chainendpoint: Optional[str] = os.getenv(core_cst.SUBTENSOR_CHAINENDPOINT_PARAM, None)

    image_worker_url: Optional[str] = os.getenv(core_cst.IMAGE_WORKER_URL_PARAM, None)
    mixtral_text_worker_url: Optional[str] = os.getenv(core_cst.MIXTRAL_TEXT_WORKER_URL_PARAM, None)
    llama_3_text_worker_url: Optional[str] = os.getenv(core_cst.LLAMA_3_TEXT_WORKER_URL_PARAM, None)

    axon_port: str = os.getenv(core_cst.AXON_PORT_PARAM, 8012)
    axon_external_ip: str = os.getenv(core_cst.AXON_EXTERNAL_IP_PARAM, "127.0.0.1")

    debug_miner: bool = os.getenv(core_cst.DEBUG_MINER_PARAM, False)


config = Config()
