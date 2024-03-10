import yaml
import os
from typing import Dict, Any, Optional
from core import constants as core_cst
from rich.prompt import Prompt
import rich

def device_processing_func(input: str):
    if "cuda" not in input:
        input = "cuda:" + input
    return input

def http_address_processing_func(input: str) -> str:
    if "http://" not in input:
        input = "http://" + input
    if input[-1] != "/":
        input = input + "/"
    return input

def bool_processing_func(input: str) -> bool:
    if input.lower() in ["true", "t", "1", "y", "yes"]:
        return True
    else:
        return False

def int_processing_func(input: str) -> Optional[int]:
    try:
        return int(input)
    except ValueError:
        return None

GLOBAL_PARAMETERS = {
    core_cst.HOTKEY_PARAM: {"default": "default", "message": "Hotkey name: "},
}

DEVICE_PARAMETERS = {
    core_cst.SAFETY_CHECKERS_PARAM: {"default": "cuda:0", "message": "Safety Checker Device ", "process_function": device_processing_func},
    core_cst.CLIP_DEVICE_PARAM: {"default": "cuda:0", "message": "CLIP Device ", "process_function": device_processing_func},
    core_cst.SCRIBBLE_DEVICE_PARAM: {"default": "cuda:0", "message": "Scribble Device ", "process_function": device_processing_func},
    core_cst.KANDINSKY_DEVICE_PARAM: {"default": "cuda:0", "message": "Kandinsky Device ", "process_function": device_processing_func},
    core_cst.SDXL_TURBO_DEVICE_PARAM: {"default": "cuda:0", "message": "SDXL Turbo Device ", "process_function": device_processing_func},
    core_cst.UPSCALE_DEVICE_PARAM: {"default": "cuda:0", "message": "Upscale Device ", "process_function": device_processing_func},
}

MISC_PARAMETERS  = {
    core_cst.WALLET_NAME_PARAM: {"default": "default", "message": "Wallet Name "},
    core_cst.SUBTENSOR_NETWORK_PARAM: {"default": "finney", "message": "Subtensor Network "},
    core_cst.SUBTENSOR_CHAINENDPOINT_PARAM: {
        "default": None,
        "message": "Subtensor Chain Endpoint ",
    },
    core_cst.IS_VALIDATOR_PARAM: {"default": "y", "message": "Is this a Validator hotkey? (y/n) ", "process_function": bool_processing_func},
}

VALIDATOR_PARAMETERS = {
    core_cst.API_SERVER_PORT_PARAM: {"default": None, "message": "API server port "},
    core_cst.CHECKING_SERVER_ADDRESS_PARAM: {"default": "http://127.0.0.1:6000/", "message": "Checking Server address ", "process_function": http_address_processing_func},
    core_cst.SAFETY_CHECKER_SERVER_ADDRESS_PARAM: {"default": "http://127.0.0.1:6001/", "message": "Safety Checker Server address ", "process_function": http_address_processing_func},
}

MINER_PARAMETERS = {
    core_cst.SOTA_PROVIDER_PARAM: {"default": None, "message": "Optional SOTA model (Currently midjourney) Provider.\nOptions:\n1: goapi\n2: imagine_slash", "process_function": int_processing_func},
    core_cst.SOTA_PROVIDER_API_KEY_PARAM: {"default": None, "message": "Optional SOTA Provider API Key: "},
    core_cst.AXON_PORT_PARAM: {"default": 8091, "message": "Axon Port: "},
    core_cst.AXON_EXTERNAL_IP_PARAM: {"default": None, "message": "Axon External IP: "},
}



gpu_assigned_dict = {}
config = {}

def handle_parameters(parameters: Dict[str, Any], hotkey: str):
    global config
    for parameter, metadata in parameters.items():
        if parameter == core_cst.HOTKEY_PARAM:
            continue
        while True:
            try:
                user_input = get_input(metadata)
                config[hotkey][parameter] = user_input
                break
            except ValueError:
                print("Invalid input, please try again.")

def get_input(parameter_metadata: Dict[str, Dict[str, Any]]) -> Any:
    message = f"[yellow]{parameter_metadata['message']}[/yellow][white](default: {parameter_metadata['default']})[/white]"

    user_input = Prompt.ask(message)
    if not user_input:
        user_input = parameter_metadata["default"]

    if parameter_metadata.get("process_function", None) is not None:
        processed_input = parameter_metadata["process_function"](user_input)
        return processed_input
    return user_input

def get_config():
    while True:
        hotkey = get_input(GLOBAL_PARAMETERS[core_cst.HOTKEY_PARAM])
        if hotkey == "":
            break

        config[hotkey] = {}

        one_gpu = get_input({"default": "y", "message": "Do you want to use a single gpu for all models (mainly for validators)? (y/n)", "process_function": bool_processing_func})
        if one_gpu:
            gpu = get_input({"default": "0", "message": "Which gpu do you want to use? ", "process_function": device_processing_func})
            config[hotkey][core_cst.SINGULAR_GPU] = gpu
        else:
            handle_parameters(DEVICE_PARAMETERS, hotkey)

        handle_parameters(MISC_PARAMETERS, hotkey)

        if config[hotkey][core_cst.IS_VALIDATOR_PARAM]:
            handle_parameters(VALIDATOR_PARAMETERS, hotkey)
        else:
            handle_parameters(MINER_PARAMETERS, hotkey)

        # Check if the user wants to add another hotkey
        add_another = input("Do you want to add another hotkey? (y/n, default n): ")

        if add_another.lower() != "y":
            break

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Now make a bash script for the miners, cos im kind
    delete_command = "pm2 delete "

    miner_start_commands = []

    miner_start_command_template = "pm2 start --name miner_{} {}"

    for hotkey, settings in config.items():

        delete_command += f"miner_{hotkey} "

        # If it's not a validator (aka miners), add the start command
        if not settings.get(core_cst.IS_VALIDATOR_PARAM, False):
            axon_port = settings.get(core_cst.AXON_PORT_PARAM)
            axon_external_ip = settings.get(core_cst.AXON_EXTERNAL_IP_PARAM)
            wallet_name = settings.get(core_cst.WALLET_NAME_PARAM)
            subtensor_network = settings.get(core_cst.SUBTENSOR_NETWORK_PARAM)
            subtensor_chain_endpoint = settings.get(core_cst.SUBTENSOR_CHAINENDPOINT_PARAM)

            netuid = 19 if subtensor_network != "test" else 51
            start_command = (
                f'mining/run_miner.py --interpreter python3 -- --axon.port {axon_port} --axon.external_ip {axon_external_ip}'
                f" --wallet.name {wallet_name} --wallet.hotkey {hotkey} --subtensor.network {subtensor_network}"
                f' --netuid {netuid} --logging.debug'
            )
            if subtensor_chain_endpoint is not None:
                start_command += f" --subtensor.chain_endpoint {subtensor_chain_endpoint}"

            miner_start_command = miner_start_command_template.format(hotkey, start_command)
            miner_start_commands.append(miner_start_command)

    all_miner_start_commands_str = "\n".join(miner_start_commands)
    bash_script = f"#!/bin/bash\n{delete_command}\n{all_miner_start_commands_str}"

    with open("start_miners.sh", "w") as f:
        f.write(bash_script)

    os.chmod("start_miners.sh", 0o700)

    rich.print("Config saved to config.yaml")
