import abc
from typing import Any, Tuple, TypeVar

import bittensor as bt
from fastapi.responses import JSONResponse

from core import tasks, utils
from mining.proxy import core_miner
from mining.proxy.core_miner import miner_requests_stats
from config.miner_config import config as miner_config
from functools import wraps

T = TypeVar("T", bound=bt.Synapse)

# Make sure this matches the class name
operation_name = "Operation"


class Operation(core_miner.CoreMiner):
    @staticmethod
    @abc.abstractmethod
    async def forward(synapse: Any) -> Any: ...

    @staticmethod
    @abc.abstractmethod
    def blacklist(synapse: Any) -> Tuple[bool, str]: ...

    @staticmethod
    @abc.abstractmethod
    def priority(synapse: Any) -> float: ...


def enforce_concurrency_limits(func):
    @wraps(func)
    async def wrapper(synapse: T, *args, **kwargs):
        """
        Applies concurrency limits to the operations
        """

        task = tasks.get_task_from_synapse(synapse)
        task_is_stream = tasks.TASK_IS_STREAM.get(task, False)

        capacity_config = utils.load_capacities(miner_config.hotkey_name)
        concurrency_groups = utils.load_concurrency_groups(miner_config.hotkey_name)
        concurrency_group_id = concurrency_groups.get(task.value, {}).get("concurrency_group_id")

        if concurrency_group_id is None:
            bt.logging.error(
                f"[BUG 🐞] Task '{task}' not in concurrency groups. "
                f"Capacity config: {capacity_config}. "
                f"Concurrency groups: {concurrency_groups}."
                f"Available tasks: {list(capacity_config.keys())}. "
            )
            synapse.axon.status_code = "429"
            synapse.dendrite.status_code = "429"
            return JSONResponse(
                status_code=int(synapse.axon.status_code),
                headers=synapse.to_headers(),
                content={"message": synapse.axon.status_message},
            )
        else:
            concurrency_group_id = str(concurrency_group_id)

        with core_miner.threading_lock:
            current_number_of_concurrent_requests = miner_requests_stats.active_requests_for_each_concurrency_group.get(
                concurrency_group_id, 0
            )
            if current_number_of_concurrent_requests < concurrency_groups[concurrency_group_id]:
                miner_requests_stats.active_requests_for_each_concurrency_group[concurrency_group_id] = (
                    current_number_of_concurrent_requests + 1
                )
            else:
                synapse.axon.status_code = "429"
                synapse.axon.status_message = "Enhance your calm bro"
                synapse.dendrite.status_message = "Enhance your calm bro"
                synapse.dendrite.status_code = "429"
                return JSONResponse(
                    status_code=synapse.axon.status_code,
                    content={"message": "Enhance your calm bro"},
                    headers=synapse.to_headers(),
                )
        try:
            return await func(synapse, *args, **kwargs)
        except Exception as e:
            bt.logging.error(e)
        finally:
            # wont work for stream since we get here straight away
            if not task_is_stream:
                miner_requests_stats.decrement_concurrency_group_from_task(task)

    return wrapper
