from abc import ABC
from typing import Tuple

import bittensor as bt


import asyncio
import threading

class BaseValidator(ABC):
    def __init__(self, dendrite: bt.dendrite, config, subtensor, wallet, timeout):
        self.dendrite = dendrite
        self.config = config
        self.subtensor = subtensor
        self.wallet = wallet
        self.timeout = timeout
        self.streaming = False
        self.device = config.neuron.device
        self.async_lock = asyncio.Lock()
        self.threading_lock = threading.Lock()
        bt.logging.info(f"Using device {self.device} for the validator")



    async def query_miner(self, axon: bt.axon, uid: int, syn: bt.Synapse) -> Tuple[int, bt.Synapse]:
        try:
            responses = await self.dendrite(
                [axon],
                syn,
                deserialize=False,
                timeout=self.timeout,
                streaming=self.streaming,
            )
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

    async def handle_response(self, uid: int, responses: list[bt.Synapse]) -> Tuple[int, bt.Synapse]:
        return uid, responses[0]
