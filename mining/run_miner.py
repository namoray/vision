import importlib
import pkgutil
import time
import tracemalloc

import bittensor as bt

from core import resource_management, utils
from mining import core_miner, operations

# For determinism

tracemalloc.start()

if __name__ == "__main__":
    miner = core_miner.CoreMiner()

    bt.logging.info("Loading all config & resources....")
    resource_management.set_hotkey_name(miner.wallet.hotkey_str)
    resource_manager_singleton = resource_management.SingletonResourceManager()
    resource_manager_singleton.load_config()
    resource_manager_singleton.load_all_resources()

    available_operations = resource_manager_singleton.get_available_operations()

    if miner.config.debug_miner:
        bt.logging.debug("Miner is in debug mode 🪳🔫")

    all_operations = list(pkgutil.iter_modules(operations.__path__))

    available_operations_module = importlib.import_module("operations.available_operations")
    available_operation_class_name = available_operations_module.operation_name
    AvailableOpclass = getattr(available_operations_module, available_operation_class_name)
    miner.attach_to_axon(AvailableOpclass.forward, AvailableOpclass.blacklist, AvailableOpclass.priority)

    if not miner.config.debug_miner:
        for operation in available_operations:
            operation_name = utils.pascal_to_snake(operation)
            module = importlib.import_module(f"operations.{operation_name}_operation")
            operation_class_name = module.operation_name

            OpClass = getattr(module, operation_class_name)
            miner.attach_to_axon(OpClass.forward, OpClass.blacklist, OpClass.priority)

    with miner as running_miner:
        while True:
            time.sleep(240)
