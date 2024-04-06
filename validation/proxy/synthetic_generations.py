import httpx
from typing import Dict, Any
from config.validator_config import config as validator_config

import bittensor as bt


async def get_synthetic_data(task: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=7) as client:
            response = await client.post(
                validator_config.external_server_url + "get-synthetic-data",
                json={"task": task},
            )
            response.raise_for_status()  # raises an HTTPError if an unsuccessful status code was received
    except httpx.RequestError as err:
        bt.logging.warning(f"Getting synthetic data error: {err.request.url!r}: {err}")
        return None
    except httpx.HTTPStatusError as err:
        bt.logging.warning(
            f"Syntehtic data error; status code {err.response.status_code} while requesting {err.request.url!r}: {err}"
        )
        return None

    try:
        response_json = response.json()
    except ValueError as e:
        bt.logging.error(f"Synthetic data Response contained invalid JSON: error :{e}")
        return None

    return response_json
