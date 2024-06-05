import enum
import json
import time
from typing import Any, Dict
import httpx
import bittensor as bt
from substrateinterface import Keypair


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

POST_TO_TAUVISION = True


def _sign_timestamp(keypair: Keypair, timestamp: float) -> str:
    return f"0x{keypair.sign(str(timestamp)).hex()}"


async def post_to_tauvision(
    data_to_post: Dict[str, Any], keypair: Keypair, data_type_to_post: DataTypeToPost, timeout: int = 10
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
            return resp
        except Exception as e:
            bt.logging.error(f"Error when posting to taovision to store score data: {e}")
