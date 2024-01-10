import abc
from typing import Any, Tuple, TypeVar

import bittensor as bt

from mining import core_miner

T = TypeVar("T", bound=bt.Synapse)

# Make sure this matches the class name
operation_name = "Operation"


class Operation(core_miner.CoreMiner):
    @staticmethod
    @abc.abstractmethod
    def forward(synapse: Any) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def blacklist(synapse: Any) -> Tuple[bool, str]:
        ...

    @staticmethod
    @abc.abstractmethod
    def priority(synapse: Any) -> float:
        ...
