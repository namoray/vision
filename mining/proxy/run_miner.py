import importlib
import time
import tracemalloc

import bittensor as bt
from core import tasks
from mining.proxy import core_miner
from config.miner_config import config

# For determinism

tracemalloc.start()



if __name__ == "__main__":

    miner = core_miner.CoreMiner()

    bt.logging.info("Loading all config & resources....")

    if config.debug_miner:
        bt.logging.debug("Miner is in debug mode 🪳🔫")

    available_operations_module = importlib.import_module(
        "operations.available_operations"
    )
    available_operation_class_name = available_operations_module.operation_name

    AvailableOpclass = getattr(
        available_operations_module, available_operation_class_name
    )
    miner.attach_to_axon(
        AvailableOpclass.forward, AvailableOpclass.blacklist, AvailableOpclass.priority
    )

    operations_supported = set()
    if not config.debug_miner:
        for task in tasks.SUPPORTED_TASKS:
            operation_module = tasks.TASKS_TO_MINER_OPERATION_MODULES[task]
            if operation_module.__name__ not in operations_supported:
                operations_supported.add(operation_module.__name__)
                operation_class = getattr(operation_module, operation_module.operation_name)
                miner.attach_to_axon(
                    getattr(operation_class, "forward"),
                    getattr(operation_class, "blacklist"),
                    getattr(operation_class, "priority"),
                )

    with miner as running_miner:
        while True:
            time.sleep(240)
