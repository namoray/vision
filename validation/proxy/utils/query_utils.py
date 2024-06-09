import asyncio
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ValidationError
from core import Task
from models import base_models, utility_models
import bittensor as bt
from validation.proxy.utils import constants as cst
from validation.models import UIDRecord, axon_uid
from core import bittensor_overrides as bto
from collections import OrderedDict
from validation.scoring import scoring_utils
import json


class UIDQueue:
    def __init__(self):
        self.uid_map: OrderedDict[str, None] = OrderedDict()

    def add_uid(self, uid: axon_uid) -> None:
        if uid not in self.uid_map:
            self.uid_map[uid] = None

    def get_uid_and_move_to_back(self) -> Optional[axon_uid]:
        if self.uid_map:
            uid, _ = self.uid_map.popitem(last=False)
            self.uid_map[uid] = None
            return uid
        return None

    def move_to_end(self, uid: axon_uid) -> None:
        if uid in self.uid_map:
            self.uid_map.pop(uid)
            self.uid_map[uid] = None

    def remove_uid(self, uid: axon_uid) -> None:
        if uid in self.uid_map:
            self.uid_map.pop(uid)


def get_formatted_response(
    resulting_synapse: base_models.BaseSynapse, initial_synapse: bt.Synapse
) -> Optional[BaseModel]:
    if resulting_synapse and resulting_synapse.dendrite.status_code == 200 and resulting_synapse != initial_synapse:
        formatted_response = _extract_response(resulting_synapse, initial_synapse)
        return formatted_response
    else:
        return None


async def query_individual_axon(
    dendrite: bt.dendrite,
    axon: bto.axon,
    axon_uid: int,
    synapse: bt.Synapse,
    deserialize: bool = False,
    log_requests_and_responses: bool = True,
) -> Tuple[base_models.BaseSynapse, float]:
    operation_name = synapse.__class__.__name__
    if operation_name not in cst.OPERATION_TIMEOUTS:
        bt.logging.warning(
            f"Operation {operation_name} not in operation_to_timeout, this is probably a mistake / bug 🐞"
        )

    start_time = time.time()

    if log_requests_and_responses:
        bt.logging.info(f"Querying axon {axon_uid} for {operation_name}")

    response = await dendrite.forward(
        axons=axon,
        synapse=synapse,
        connect_timeout=1.0,
        response_timeout=cst.OPERATION_TIMEOUTS.get(operation_name, 15),
        deserialize=deserialize,
        log_requests_and_responses=log_requests_and_responses,
        streaming=False,
    )
    return response, time.time() - start_time


def _load_sse_jsons(chunk: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    try:
        jsons = []
        received_event_chunks = chunk.split("\n\n")
        for event in received_event_chunks:
            if event == "":
                continue
            prefix, _, data = event.partition(":")
            if data.strip() == "[DONE]":
                break
            loaded_chunk = json.loads(data)
            jsons.append(loaded_chunk)
        return jsons
    except json.JSONDecodeError:
        try:
            loaded_chunk = json.loads(chunk)
            if "message" in loaded_chunk:
                return {
                    "message": loaded_chunk["message"],
                    "status_code": "429" if "bro" in loaded_chunk["message"] else "500",
                }
        except json.JSONDecodeError:
            ...

    return []


def _get_formatted_payload(content: str, first_message: bool, add_finish_reason: bool = False) -> str:
    delta_payload = {"content": content}
    if first_message:
        delta_payload["role"] = "assistant"
    choices_payload = {"delta": delta_payload}
    if add_finish_reason:
        choices_payload["finish_reason"] = "stop"
    payload = {
        "choices": [choices_payload],
    }

    dumped_payload = json.dumps(payload)
    return dumped_payload


async def query_miner_stream(
    uid_record: UIDRecord,
    synapse: bt.Synapse,
    outgoing_model: BaseModel,
    task: Task,
    dendrite: bto.dendrite,
    synthetic_query: bool,
) -> AsyncIterator[str]:
    axon_uid = uid_record.axon_uid
    axon = uid_record.axon

    time1 = time.time()
    text_generator = await query_individual_axon_stream(
        synapse=synapse, dendrite=dendrite, axon=axon, axon_uid=axon_uid, log_requests_and_responses=False
    )
    text_jsons = []
    status_code = 200
    error_message = None
    if text_generator is not None:
        first_message = True
        async for text in text_generator:
            if isinstance(text, str):
                try:
                    loaded_jsons = _load_sse_jsons(text)
                    if isinstance(loaded_jsons, dict):
                        status_code = loaded_jsons.get("status_code")
                        error_message = loaded_jsons.get("message")
                        break

                except (IndexError, json.JSONDecodeError) as e:
                    bt.logging.warning(f"Error {e} when trying to load text: {text}")
                    break

                text_jsons.extend(loaded_jsons)
                for text_json in loaded_jsons:
                    content = text_json.get("text", "")
                    if content == "":
                        continue
                    dumped_payload = _get_formatted_payload(content, first_message)
                    first_message = False
                    yield f"data: {dumped_payload}\n\n"

        if len(text_jsons) > 0:
            last_payload = _get_formatted_payload("", False, add_finish_reason=True)
            yield f"data: {last_payload}\n\n"
            yield "data: [DONE]\n\n"
            bt.logging.info(f"✅ Successfully queried axon: {axon_uid} for task: {task}")

        response_time = time.time() - time1
        query_result = utility_models.QueryResult(
            formatted_response=text_jsons if len(text_jsons) > 0 else None,
            axon_uid=axon_uid,
            response_time=response_time,
            task=task,
            success=not first_message,
            miner_hotkey=uid_record.hotkey,
            status_code=status_code,
            error_message=error_message,
        )

        create_scoring_adjustment_task(query_result, synapse, uid_record, synthetic_query)


def create_scoring_adjustment_task(
    query_result: utility_models.QueryResult, synapse: bt.Synapse, uid_record: UIDRecord, synthetic_query: bool
):
    asyncio.create_task(
        scoring_utils.adjust_uid_record_from_result(query_result, synapse, uid_record, synthetic_query=synthetic_query)
    )


async def query_miner_no_stream(
    uid_record: UIDRecord,
    synapse: bt.Synapse,
    outgoing_model: BaseModel,
    task: Task,
    dendrite: bto.dendrite,
    synthetic_query: bool,
) -> utility_models.QueryResult:
    axon_uid = uid_record.axon_uid
    axon = uid_record.axon
    resulting_synapse, response_time = await query_individual_axon(
        synapse=synapse, dendrite=dendrite, axon=axon, axon_uid=axon_uid, log_requests_and_responses=False
    )

    # IDE doesn't recognise the above typehints, idk why? :-(
    resulting_synapse: base_models.BaseSynapse
    response_time: float

    formatted_response = get_formatted_response(resulting_synapse, outgoing_model)
    if formatted_response is not None:
        bt.logging.info(f"✅ Successfully queried axon: {axon_uid} for task: {task}")
        query_result = utility_models.QueryResult(
            formatted_response=formatted_response,
            axon_uid=axon_uid,
            response_time=response_time,
            task=task,
            success=True,
            miner_hotkey=uid_record.hotkey,
            status_code=resulting_synapse.axon.status_code,
            error_message=resulting_synapse.error_message,
        )
        create_scoring_adjustment_task(query_result, synapse, uid_record, synthetic_query)
        return query_result

    query_result = utility_models.QueryResult(
        formatted_response=None,
        axon_uid=axon_uid,
        response_time=None,
        error_message=resulting_synapse.axon.status_message,
        task=task,
        status_code=resulting_synapse.axon.status_code,
        success=False,
        miner_hotkey=uid_record.hotkey,
    )
    create_scoring_adjustment_task(query_result, synapse, uid_record, synthetic_query)
    return query_result


def _extract_response(resulting_synapse: base_models.BaseSynapse, outgoing_model: BaseModel) -> Optional[BaseModel]:
    try:
        formatted_response = outgoing_model(**resulting_synapse.dict())

        # If we're expecting a result (i.e. not nsfw), then try to deserialize
        if (hasattr(formatted_response, "is_nsfw") and not formatted_response.is_nsfw) or not hasattr(
            formatted_response, "is_nsfw"
        ):
            deserialized_result = resulting_synapse.deserialize()
            if deserialized_result is None:
                formatted_response = None

        return formatted_response
    except ValidationError as e:
        bt.logging.debug(f"Failed to deserialize for some reason: {e}")
        return None


async def query_individual_axon_stream(
    dendrite: bt.dendrite,
    axon: bto.axon,
    axon_uid: int,
    synapse: bt.Synapse,
    deserialize: bool = False,
    log_requests_and_responses: bool = True,
):
    synapse_name = synapse.__class__.__name__
    if synapse_name not in cst.OPERATION_TIMEOUTS:
        bt.logging.warning(f"Operation {synapse_name} not in operation_to_timeout, this is probably a mistake / bug 🐞")
    if log_requests_and_responses:
        bt.logging.info(f"Querying axon {axon_uid} for {synapse_name}")
    response = await dendrite.forward(
        axons=axon,
        synapse=synapse,
        connect_timeout=0.3,
        response_timeout=5,  # if X seconds without any data, its boinked
        deserialize=deserialize,
        log_requests_and_responses=log_requests_and_responses,
        streaming=True,
    )
    return response
