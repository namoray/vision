
import argparse
import pathlib
from typing import Dict, Any, Optional
import bittensor as bt
from core import constants as core_cst

def check_config(config: "bt.Config") -> None:
    bt.axon.check_config(config)
    bt.logging.check_config(config)

    config.miner.full_path = (
        pathlib.Path(config.logging.logging_dir)
        .joinpath(config.wallet.get("name", bt.defaults.wallet.name))
        .joinpath(config.wallet.get("hotkey", bt.defaults.wallet.hotkey))
        .joinpath(config.miner.name)
        .expanduser()
        .absolute()
    )
    config.miner.full_path.mkdir(parents=True, exist_ok=True)

def get_miner_cli_config() -> "bt.Config":

    parser = argparse.ArgumentParser()


    parser.add_argument("--axon.port", type=int, default=8091, help="Port to run the axon on.")

    parser.add_argument("--axon.external_ip", type=str, default=None)

    parser.add_argument(
        "--debug_miner", action="store_true", default=False
    )

    parser.add_argument(
        "--subtensor.network",
        default="finney",
        help="Bittensor network to connect to.",
    )

    parser.add_argument(
        "--subtensor.chain_endpoint",
        default=None,
        help="Chain endpoint to connect to.",
    )

    parser.add_argument("--netuid", type=int, default=19, help="The chain subnet uid.")

    parser.add_argument("--wallet.name", type=str, default="default")
    parser.add_argument("--wallet.hotkey", type=str, default="default")

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)

    config = bt.config(parser)

    config.full_path = (
        pathlib.Path(config.logging.logging_dir)
        .joinpath(config.wallet.name)
        .joinpath(config.wallet.hotkey)
        .joinpath("netuid{}".format(config.netuid))
        .joinpath("miner")
    )
    config.full_path.mkdir(parents=True, exist_ok=True)
    return config

def get_validator_cli_config(yaml_config: Dict[str, Any], hotkey_name: Optional[str] = None) -> "bt.Config":
    parser = argparse.ArgumentParser()


    subtensor_network = yaml_config.get(core_cst.SUBTENSOR_NETWORK_PARAM, "finney")
    parser.add_argument(
        "--subtensor.network",
        default=subtensor_network,
        help="Bittensor network to connect to.",
    )

    parser.add_argument(
        "--subtensor.chain_endpoint",
        default=yaml_config.get(core_cst.SUBTENSOR_CHAINENDPOINT_PARAM, None),
        help="Chain endpoint to connect to.",
    )

    parser.add_argument("--netuid", type=int, default=19 if subtensor_network != "test" else 51 , help="The chain subnet uid.")

    parser.add_argument("--debug_miner", action="store_true", default=False)

    parser.add_argument("--wallet.name", type=str, default=yaml_config.get(core_cst.WALLET_NAME_PARAM, "default"))
    parser.add_argument("--wallet.hotkey", type=str, default=hotkey_name if hotkey_name is not None else "default")

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)

    config = bt.config(parser)

    config.full_path = (
        pathlib.Path(config.logging.logging_dir)
        .joinpath(config.wallet.name)
        .joinpath(config.wallet.hotkey)
        .joinpath("netuid{}".format(config.netuid))
        .joinpath("miner")
    )
    config.full_path.mkdir(parents=True, exist_ok=True)
    return config
