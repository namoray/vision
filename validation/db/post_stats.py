from core import Task
import enum
import json
import time
from typing import Any, Dict, List, Union
import httpx
import bittensor as bt
from pydantic import BaseModel
from substrateinterface import Keypair

from validation.models import RewardData


class DataTypeToPost(enum.Enum):
    REWARD_DATA = 1
    UID_RECORD = 2
    MINER_CAPACITIES = 3
    VALIDATOR_INFO = 4


BASE_URL = "https://dev.tauvision.ai/"

data_type_to_url = {
    DataTypeToPost.REWARD_DATA: BASE_URL + "v1/store/reward_data",
    DataTypeToPost.UID_RECORD: BASE_URL + "v1/store/uid_record",
    DataTypeToPost.MINER_CAPACITIES: BASE_URL + "v1/store/miner_capacities",
    DataTypeToPost.VALIDATOR_INFO: BASE_URL + "v1/store/validator_info",
}

# Turn off if you don't wanna post your validator info to tauvision

POST_TO_TAUVISION = False


def _sign_timestamp(keypair: Keypair, timestamp: float) -> str:
    return f"0x{keypair.sign(str(timestamp)).hex()}"


async def post_to_tauvision(
    data_to_post: Union[Dict[str, Any], List[Dict[str, Any]]],
    keypair: Keypair,
    data_type_to_post: DataTypeToPost,
    timeout: int = 10,
) -> None:
    if not POST_TO_TAUVISION:
        return
    timestamp = time.time()
    public_address = keypair.ss58_address
    signed_timestamp = _sign_timestamp(keypair, timestamp)

    headers = {
        "Content-Type": "application/json",
        "x-timestamp": str(timestamp),
        "x-signature": signed_timestamp,
        "x-public-key": public_address,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                url=data_type_to_url[data_type_to_post],
                data=json.dumps(data_to_post),
                headers=headers,
            )
            bt.logging.debug(f"Resp from taovision: {resp.text} for post type {data_type_to_post}")
            return resp
        except Exception as e:
            bt.logging.error(f"Error when posting to taovision to store score data: {e}")


class RewardDataPostBody(RewardData):
    testnet: bool


class ValidatorInfoPostBody(BaseModel):
    versions: str
    validator_hotkey: str


class MinerCapacitiesPostObject(BaseModel):
    validator_hotkey: str
    miner_hotkey: str
    task: Task
    volume: float


class MinerCapacitiesPostBody(BaseModel):
    data: List[MinerCapacitiesPostObject]

    def dump(self):
        return [
            {
                "validator_hotkey": ob.validator_hotkey,
                "miner_hotkey": ob.miner_hotkey,
                "task": ob.task.value,
                "volume": ob.volume,
            }
            for ob in self.data
        ]


class UidRecordPostBody(BaseModel):
    axon_uid: int
    miner_hotkey: str
    validator_hotkey: str
    task: Task
    declared_volume: float
    consumed_volume: float
    total_requests_made: int
    requests_429: int
    requests_500: int
    period_score: int

    def dict(self):
        return {
            "axon_uid": self.axon_uid,
            "miner_hotkey": self.miner_hotkey,
            "validator_hotkey": self.validator_hotkey,
            "task": self.task.value,
            "declared_volume": self.declared_volume,
            "consumed_volume": self.consumed_volume,
            "total_requests_made": self.total_requests_made,
            "requests_429": self.requests_429,
            "requests_500": self.requests_500,
            "period_score": self.period_score,
        }
