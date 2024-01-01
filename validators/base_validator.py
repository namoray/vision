import asyncio
import threading
from abc import ABC
from typing import Tuple
import markovify
from datasets import load_dataset
import bittensor as bt
from dotenv import load_dotenv

load_dotenv()

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

        dataset = load_dataset("multi-train/coco_captions_1107")
        text = [i["query"] for i in dataset["train"]]
        self.markov_text_generation_model = markovify.Text(" ".join(text))


    async def query_miner(self, axon: bt.axon, uid: int, syn: bt.Synapse) -> Tuple[int, bt.Synapse]:
        try:
            dendrite = bt.dendrite(self.wallet)
            responses = await dendrite(
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
