import asyncio
import copy
import random
import re
import threading
import time
from collections import defaultdict, deque
from typing import Dict
from typing import List, Any
from typing import AsyncIterator, AsyncGenerator
from typing import Optional
from typing import Set
from typing import Tuple
from core import tasks
import bittensor as bt
import httpx
import torch
from pydantic import BaseModel
from pydantic import ValidationError
from core import bittensor_overrides as bto
from config import configuration
from config.validator_config import config as validator_config
from core import constants as core_cst
from models import base_models
from models import synapses
from models import utility_models
from validation.proxy import constants as cst
from validation.proxy import validation_utils
from validation.proxy.db import DatabaseManager
from validation.proxy import synthetic_generations
import json
import traceback

db_manager = DatabaseManager()

VERSION_KEY = 20_004


_PASCAL_SEP_REGEXP = re.compile("(.)([A-Z][a-z]+)")
_UPPER_FOLLOWING_REGEXP = re.compile("([a-z0-9])([A-Z])")
MAX_PERIODS_TO_LOOK_FOR_SCORE = 16


def _pascal_to_kebab(input_string: str) -> str:
    hyphen_separated = _PASCAL_SEP_REGEXP.sub(r"\1-\2", input_string)
    return _UPPER_FOLLOWING_REGEXP.sub(r"\1-\2", hyphen_separated).lower()


async def _store_result_in_sql_lite_db(
    result: utility_models.QueryResult, task: str, synapse: bt.Synapse, synthetic_query: bool
) -> None:
    result_in_dict_form = result.dict()
    db_manager.insert_task_results(task, result_in_dict_form, synapse, synthetic_query)


def _load_sse_jsons(chunk: str) -> List[Dict[str, Any]]:
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


class CoreValidator:
    def __init__(self) -> None:
        self.config = self.prepare_config_and_logging()
        self.subtensor = bt.subtensor(config=self.config)
        self.wallet = bt.wallet(config=self.config)
        self.keypair = self.wallet.hotkey
        self.dendrite = bto.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid, lite=True)

        validation_utils.connect_to_external_server()

        # Make the above class variables instead

        bt.logging(debug=True)

        self.uids: list[int] = []
        self.axon_indexes: list[int] = []
        self.incentives: list[float] = []
        self.axons: list[bt.axon] = []

        self.uid_to_uid_info: Dict[int, utility_models.UIDinfo] = {}
        self.previous_uid_infos: deque[List[utility_models.UIDinfo]] = deque([], maxlen=MAX_PERIODS_TO_LOOK_FOR_SCORE)

        self.low_incentive_uids: Set[int] = set()

        self.tasks_to_available_axon_uids: Dict[str, Set[int]] = defaultdict(lambda: set())
        self.failed_queries_per_miner: Dict[int, int] = defaultdict(lambda: 0)

        self.threading_lock = threading.Lock()

        self.results_store: Dict[str, utility_models.QueryResult] = {}

    def prepare_config_and_logging(self) -> bt.config:
        base_config = configuration.get_validator_cli_config()

        bt.logging(config=base_config, logging_dir=base_config.full_path)
        return base_config

    def start_continuous_tasks(self):
        self.resync_task = asyncio.create_task(self.periodically_resync_and_set_weights())
        self.resync_task.add_done_callback(validation_utils.log_task_exception)

        self.score_task = asyncio.create_task(self.synthetically_score())
        self.score_task.add_done_callback(validation_utils.log_task_exception)

        self.score_results_task = asyncio.create_task(self.score_results())
        self.score_results_task.add_done_callback(validation_utils.log_task_exception)

    async def _resync_metagraph_and_sleep(self, time_between_resyncing: float, set_weights: bool) -> None:
        await self.resync_metagraph()
        if set_weights:
            await asyncio.to_thread(self.set_weights)
        await asyncio.sleep(time_between_resyncing)

    async def periodically_resync_and_set_weights(self) -> None:
        # TODO: CHANGE AFTER DEBUGING
        cycle_length_initial = 0
        cycle_length_in_loop = 1
        time_between_resyncing = 60 * 30  # 30 mins

        # Initial cycles to make sure restarts don't impact scores too heavily
        for _ in range(cycle_length_initial):
            await self._resync_metagraph_and_sleep(time_between_resyncing, set_weights=False)

        while True:
            for _ in range(cycle_length_in_loop):
                await self._resync_metagraph_and_sleep(time_between_resyncing, set_weights=False)

            await self._resync_metagraph_and_sleep(time_between_resyncing, set_weights=True)

    async def _store_synthetic_result(self, sota_request: utility_models.SotaCheckingRequest) -> None:
        """
        Sends to taovision (subnet 19 api) the image url and prompt to store for later training.

        Signed to authenticate using the same mechanism that dendrites use to send requests.
        First, sign the message using the keypair, then send off the signed message (encrypted) and the
        public key (which is just the hotkey ss58_address)

        Then use the public key to decrypt the message - if it's from a validator then on subnet 19 then accept it,
        else don't (so only validators can store images into the dataset)

        This is entirely copied from the preprocess_synapse_for_request method
        in the bittensor dendrite class - which is utilised for every synapse request. I.e. this
        information is already sent to every single miner you interact with
        """
        time_to_sign = time.time()
        string_time_to_sign = str(time_to_sign)
        time_signed = f"0x{self.keypair.sign(string_time_to_sign).hex()}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                await client.post(
                    url="https://taovision.ai/store-sota-image",
                    json={
                        "image_url": sota_request.image_url,
                        "timestamp": string_time_to_sign,
                        "signed_message": time_signed,
                        "hotkey_public_address": self.keypair.ss58_address,
                        "prompt": sota_request.prompt,
                    },
                )
        except Exception as e:
            bt.logging.debug(
                f"Error when storing sota image: {e}. Doesn't matter too much, unless this repeatedly happens"
            )

    async def _query_checking_server_for_synthetic_data(self, operation: str) -> Optional[utility_models.QueryResult]:
        endpoint = _pascal_to_kebab(operation)
        url = validator_config.external_server_url + endpoint
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.get(url)

            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError as e:
            error_details = str(e)
            traceback_info = traceback.format_exc()
            bt.logging.debug(
                f"Error when querying endpoint {url}: {error_details}\n Traceback information: {traceback_info}"
            )
            return None

    async def synthetically_score(self) -> None:
        """
        Asynchronously scores axons synthetically.

        This function continuously picks random axon UIDs that have organic scores of 0
        and calculates synthetic scores for them. The function checks if there are any
        axon UIDs that have both organic scores and synthetic scores of 0. If there are,
        it randomly selects one of those UIDs. Otherwise, it randomly selects one of the
        axon UIDs with organic scores of 0.

        The function then retrieves the operations available for the selected axon UID.
        It filters out the operations that are not "AvailableOperations" or "Synapse".
        If there are any operations available, the function measures the time before
        scoring and calls the 'synthetically_score_axon' function to calculate the
        synthetic scores for the selected axon UID. After scoring, the function waits for
        a certain amount of time before moving on to the next iteration.

        If there are no operations available, the function waits for a longer duration
        before moving on to the next iteration.

        This function does not return any value.
        """
        return
        while True:
            # TODO: mimic taovision when we're live
            task = random.choice(list(tasks.TASKS_TO_MINER_OPERATION_MODULES.keys()))

            # TEMP
            if task == tasks.Tasks.avatar.value:
                continue
            #

            # We don't want to put too much emphasis on sota, so query it a lot less
            if task == tasks.Tasks.sota.value:
                if random.random() > 0.03:
                    continue
            synthetic_data = await synthetic_generations.get_synthetic_data(task)
            if synthetic_data is None:
                bt.logging.debug(
                    f"Synthetic data is none for operation {task}, THIS IS NORMAL AS THIS HAPPENS PERIODICALLY,  will try again in {cst.MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING}."
                )
                await asyncio.sleep(cst.MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING)
                continue
            bt.logging.info(f"🔥 got synthetic data for task {task}")
            synthetic_synapse = tasks.TASKS_TO_SYNAPSE[task](**synthetic_data)
            stream = isinstance(synthetic_synapse, bt.StreamingSynapse)

            outgoing_model = getattr(base_models, synthetic_synapse.__class__.__name__ + core_cst.OUTGOING)

            time_before_query = time.time()

            asyncio.create_task(
                self.execute_query(
                    synapse=synthetic_synapse,
                    outgoing_model=outgoing_model,
                    synthetic_query=True,
                    task=task,
                    stream=stream,
                )
            )

            time_to_execute_query = time.time() - time_before_query
            await asyncio.sleep(
                max(
                    cst.MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING - time_to_execute_query,
                    0,
                )
            )

    async def score_results(self):
        while True:
            if len(self.uid_to_uid_info) == 0:
                await asyncio.sleep(5)
                continue

            tasks_and_number_of_results = db_manager.get_tasks_and_number_of_results()
            total_tasks_stored = sum(tasks_and_number_of_results.values())

            # TODO: Review if this is a good number
            if total_tasks_stored < cst.MINIMUM_TASKS_TO_START_SCORING:
                await asyncio.sleep(5)
                continue

            else:
                task_to_score = random.choices(
                    list(tasks_and_number_of_results.keys()), weights=list(tasks_and_number_of_results.values()), k=1
                )[0]
                await self._check_score_for_task(task_to_score)
                await asyncio.sleep(5)

    async def _check_score_for_task(self, task: str) -> None:
        i = 0
        while i < cst.MAX_RESULTS_TO_SCORE_FOR_TASK:
            checking_data = db_manager.select_and_delete_task_result(task)  # noqa
            if checking_data is None:
                return
            results, synthetic_query, synapse_dict_str = (
                checking_data["result"],
                checking_data["synthetic_query"],
                checking_data["synapse"],
            )
            results_json = json.loads(results)

            synapse = json.loads(synapse_dict_str)

            data = {
                "synapse": synapse,
                "synthetic_query": synthetic_query,
                "result": results_json,
                "task": task,
            }
            async with httpx.AsyncClient(timeout=180) as client:
                try:
                    response = await client.post(
                        validator_config.external_server_url + "check-result",
                        data=json.dumps(data),
                    )
                    response.raise_for_status()

                except httpx.HTTPStatusError as stat_err:
                    bt.logging.error(f"When scoring, HTTP error occurred: {stat_err}")
                    continue

                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout) as read_err:
                    bt.logging.error(f"When scoring, Read timeout occurred: {read_err}")
                    continue

                except httpx.HTTPError as http_err:
                    bt.logging.error(f"When scoring, HTTP error occurred: {http_err}")
                    continue

            try:
                response_json = response.json()
                axon_scores = response_json["axon_scores"]
            except (json.JSONDecodeError, KeyError) as parse_err:
                bt.logging.error(f"Error occurred when parsing the response: {parse_err}")
                continue

            max_expected_score = await validation_utils.get_expected_score(
                utility_models.QueryResult(**results_json), synapse, task
            )
            bt.logging.info(
                f"Adding scores: {axon_scores}; for task {task} with max expected score to normalise with {max_expected_score}"
            )
            for uid, score in axon_scores.items():
                uid_info = self.uid_to_uid_info[int(uid)]
                uid_info.add_score(score / max_expected_score)
            i += 1

            ## Renabled soon, this is to enable beautiful stats for the network

            # task_uuid = str(uuid.uuid4())
            # timestamp = time.time()
            # for uid, score in axon_scores.items():
            #     data_to_post = {
            #         "uid": int(uid),
            #         "score": score,
            #         "request_id": task_uuid,
            #         "task": task,
            #         "valid_response": score > cst.FAILED_RESPONSE_SCORE,
            #         "response_time": results_json["response_time"] if score != 0 else None,
            #         "timestamp": timestamp,
            #         "validator_hotkey": self.keypair.ss58_address,
            #         "synthetic_query": synthetic_query,
            #         "miner_hotkey": self.uid_to_uid_info[int(uid)].hotkey,
            #         "testnet": validator_config.subtensor_network == "test",
            #     }
            # bt.logging.info("Posting to taovision: " + json.dumps(data_to_post))

            # Post to taovision
            # async with httpx.AsyncClient(timeout=180) as client:
            #     try:
            #         await client.post(
            #             url="https://taovision.ai/store_score_data",
            #             data=json.dumps(data_to_post),
            #         )
            #     except Exception as e:
            #         bt.logging.error(f"Error when posting to taovision to store score data: {e}")

    async def fetch_available_tasks_for_each_axon(self) -> None:
        uid_to_query_task = {}

        for uid in self.uid_to_uid_info.keys():
            task = asyncio.create_task(
                self.query_individual_axon(
                    synapse=synapses.AvailableTasksOperation(),
                    axon_uid=uid,
                    deserialize=True,
                    log_requests_and_responses=True,
                )
            )
            uid_to_query_task[uid] = task

        responses_and_response_times: List[Tuple[Optional[List[str]], float]] = await asyncio.gather(
            *uid_to_query_task.values()
        )

        uids = uid_to_query_task.keys()
        all_available_tasks = [i[0] for i in responses_and_response_times]

        for uid, available_tasks in zip(uids, all_available_tasks):
            if available_tasks is None:
                continue

            tasks_for_uid = []
            allowed_tasks = set([task.value for task in tasks.Tasks])
            for task_name in available_tasks:
                # This is to stop people claiming tasks that don't exist
                if task_name not in allowed_tasks:
                    continue
                self.tasks_to_available_axon_uids[task_name].add(uid)
                tasks_for_uid.append(task_name)

            self.uid_to_uid_info[uid].available_tasks = tasks_for_uid
            bt.logging.debug(f"{uid} has available tasks: {tasks_for_uid}")
        bt.logging.info("Done fetching available tasks!")

    async def resync_metagraph(self):
        """
        Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph.
        This really needs to work in a separate runtime environment, or can query an api directly as a first try
        """
        bt.logging.info("Resyncing the metagraph!")

        self.previous_uid_infos.append(copy.deepcopy([uid_info for _, uid_info in self.uid_to_uid_info.items()]))

        await asyncio.to_thread(self.metagraph.sync, subtensor=self.subtensor, lite=True)

        bt.logging.info("Done syncing, now just extracting the valuable info")
        incentives_tensor, axon_indexes_tensor = self.metagraph.incentive.sort(descending=True)

        with self.threading_lock:
            self.uid_to_uid_info = {}
            self.uids: List[int] = self.metagraph.uids.tolist()
            self.axon_indexes = axon_indexes_tensor.tolist()
            self.incentives = incentives_tensor.tolist()
            hotkeys: List[str] = self.metagraph.hotkeys
            self.axons = self.metagraph.axons

            for i in self.axon_indexes:
                uid = self.uids[i]
                self.uid_to_uid_info[uid] = utility_models.UIDinfo(
                    uid=uid,
                    axon=self.axons[i],
                    incentive=self.incentives[i],
                    hotkey=hotkeys[i],
                )

            incentives_with_uids = list(zip(self.incentives, self.uids))
            non_zero_incentives_with_uids = [item for item in incentives_with_uids if item[0] > 0]
            sorted_incentives_with_uids = sorted(non_zero_incentives_with_uids, key=lambda x: x[0])

            low_incentive_cutoff_index = int(
                len(sorted_incentives_with_uids) * (cst.BOTTOM_PERCENTAGE_OF_MINERS_TO_IGNORE)
            )

            if low_incentive_cutoff_index >= 4:
                self.low_incentive_uids = set(
                    uid for incentive, uid in sorted_incentives_with_uids[:low_incentive_cutoff_index]
                )
                for uid in self.low_incentive_uids:
                    self.uid_to_uid_info[uid].is_low_incentive = True
            else:
                self.low_incentive_uids = set()

        bt.logging.info("Finished extraction - now to fetch the available operations for each axon")
        await self.fetch_available_tasks_for_each_axon()

        return

    async def get_generator_from_individual_axon(
        self,
        synapse: bt.Synapse,
        axon_uid: int,
        deserialize: bool = False,
        log_requests_and_responses: bool = True,
    ):
        synapse_name = synapse.__class__.__name__
        if synapse_name not in cst.OPERATION_TIMEOUTS:
            bt.logging.warning(
                f"Operation {synapse_name} not in operation_to_timeout, this is probably a mistake / bug 🐞"
            )

        bt.logging.info(f"Querying axon {axon_uid} for {synapse_name}, axon: {self.uid_to_uid_info[axon_uid].axon}, ")
        response = await self.dendrite.forward(
            axons=self.uid_to_uid_info[axon_uid].axon,
            synapse=synapse,
            connect_timeout=0.3,
            response_timeout=2,  # if X seconds without any data, its boinked
            deserialize=deserialize,
            log_requests_and_responses=log_requests_and_responses,
            streaming=True,
        )
        return response

    async def query_individual_axon(
        self,
        synapse: bt.Synapse,
        axon_uid: int,
        deserialize: bool = False,
        log_requests_and_responses: bool = True,
    ) -> Tuple[base_models.BaseSynapse, float]:
        operation_name = synapse.__class__.__name__
        if operation_name not in cst.OPERATION_TIMEOUTS:
            bt.logging.warning(
                f"Operation {operation_name} not in operation_to_timeout, this is probably a mistake / bug 🐞"
            )

        start_time = time.time()

        bt.logging.info(f"Querying axon {axon_uid} for {operation_name}")

        response = await self.dendrite.forward(
            axons=self.uid_to_uid_info[axon_uid].axon,
            synapse=synapse,
            connect_timeout=1.0,
            response_timeout=cst.OPERATION_TIMEOUTS.get(operation_name, 15),
            deserialize=deserialize,
            log_requests_and_responses=log_requests_and_responses,
            streaming=False,
        )
        return response, time.time() - start_time

    async def execute_query(
        self,
        synapse: bt.Synapse,
        task: str,
        outgoing_model: BaseModel,
        synthetic_query: bool = False,
        stream: bool = False,
    ) -> utility_models.QueryResult:
        available_axons = self._get_available_axons(task)
        if not available_axons:
            bt.logging.warning(f"No axons available for query for {task}")
            return utility_models.QueryResult(error_message=f"No axons available for operation {task}")

        should_score = synthetic_query or random.random() < cst.SCORE_QUERY_PROBABILITY

        miners_to_query_order = self._get_miners_query_order(available_axons)

        if stream:
            main_query_result = self._stream_response_from_stream_miners_until_result(
                miners_to_query_order, synapse, task, should_score, synthetic_query
            )
            if synthetic_query:
                async for chunk in main_query_result:
                    ...
        else:
            main_query_result = await self._query_miners_until_result(
                miners_to_query_order, synapse, outgoing_model, task, should_score, synthetic_query
            )

        return main_query_result

    def _get_available_axons(self, task_name: str) -> List[int]:
        return list(self.tasks_to_available_axon_uids.get(task_name, []))

    def _get_formatted_payload(self, content: str, first_message: bool, add_finish_reason: bool = False) -> str:
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

    def _get_miners_query_order(self, available_axons: List[int]) -> list:
        random.shuffle(available_axons)
        return available_axons

    async def _get_text_from_text_generator(
        self,
        text_generator: AsyncIterator[str],
    ) -> str:
        text_buffer = ""
        async for text in text_generator:
            if isinstance(text, str):
                text_buffer += text
        return text_buffer

    async def _stream_response_from_stream_miners_until_result(
        self,
        miners_to_query_order: list,
        synapse: bt.Synapse,
        task: str,
        should_score: bool,
        synthetic_query: bool,
    ) -> AsyncGenerator[str, None]:
        failed_axon_uids = []
        internal_server_errors = 0

        for axon_uid in miners_to_query_order:
            bt.logging.debug(f"Querying streaming axon: {axon_uid} for synapse {synapse.__class__.__name__}")

            time1 = time.time()
            text_generator = await self.get_generator_from_individual_axon(synapse, axon_uid)
            if text_generator is not None:
                text_jsons = []

                first_message = True
                async for text in text_generator:
                    if isinstance(text, str):
                        try:
                            loaded_jsons = _load_sse_jsons(text)
                        except IndexError as e:
                            bt.logging.warning(f"Error {e} when trying to load text {text}")
                            break

                        text_jsons.extend(loaded_jsons)
                        for text_json in loaded_jsons:
                            content = text_json.get("text", "")
                            if content == "":
                                continue
                            dumped_payload = self._get_formatted_payload(content, first_message)
                            first_message = False
                            yield f"data: {dumped_payload}\n\n"

                if len(text_jsons) > 0:
                    last_payload = self._get_formatted_payload("", False, add_finish_reason=True)
                    yield f"data: {last_payload}\n\n"
                    yield "data: [DONE]\n\n"
                    if should_score:
                        bt.logging.info(f"✅ Successfully queried axon: {axon_uid} for task: {task}")
                        query_result = utility_models.QueryResult(
                            formatted_response=text_jsons,
                            axon_uid=axon_uid,
                            failed_axon_uids=failed_axon_uids,
                            response_time=time.time() - time1,
                            error_message=None,
                        )
                        await _store_result_in_sql_lite_db(
                            result=query_result, task=task, synapse=synapse, synthetic_query=synthetic_query
                        )
                    break

            internal_server_errors += 1
            failed_axon_uids.append(axon_uid)
            if internal_server_errors >= cst.MAX_INTERNAL_SERVER_ERRORS:
                bt.logging.warning(
                    " ❌ Too many internal server errors, something is wrong with your request, unable to get a valid response from any axon :("
                )
                return

    async def _query_miners_for_result(
        self, miners_to_query_order: list, synapse: bt.Synapse, outgoing_model: BaseModel, task: str
    ) -> utility_models.QueryResult:
        internal_server_errors = 0
        failed_axon_uids = []

        for axon_uid in miners_to_query_order:
            resulting_synapse, response_time = await self.query_individual_axon(synapse, axon_uid)

            formatted_response = self._get_formatted_response(resulting_synapse, outgoing_model)
            if formatted_response is not None:
                bt.logging.info(f"✅ Successfully queried axon: {axon_uid} for task: {task}")
                return utility_models.QueryResult(
                    formatted_response=formatted_response,
                    axon_uid=axon_uid,
                    failed_axon_uids=failed_axon_uids,
                    response_time=response_time,
                    error_message=resulting_synapse.error_message,
                )

            bt.logging.debug(f"Failed response from axon: {axon_uid} for {synapse.__class__.__name__} :(")

            internal_server_errors += 1
            failed_axon_uids.append(axon_uid)
            if internal_server_errors >= cst.MAX_INTERNAL_SERVER_ERRORS:
                bt.logging.debug("Too many internal server errors, something is wrong with the request :/")
                return utility_models.QueryResult(
                    error_message=resulting_synapse.error_message,
                    failed_axon_uids=[],
                    axon_uid=axon_uid,
                )

        bt.logging.warning(f"❌ Unable to get a valid response from any axon :( for task {task}")
        return utility_models.QueryResult(
            error_message="Unable to get a valid response from any axon",
            failed_axon_uids=failed_axon_uids,
            axon_uid=axon_uid,
        )

    async def _query_miners_until_result(
        self,
        miners_to_query_order: list,
        synapse: bt.Synapse,
        outgoing_model: BaseModel,
        task: str,
        should_score: bool,
        synthetic_query: bool,
    ) -> utility_models.QueryResult:
        result = await self._query_miners_for_result(miners_to_query_order, synapse, outgoing_model, task)
        if should_score:
            await _store_result_in_sql_lite_db(result, task, synapse, synthetic_query)
        return result

    def _get_formatted_response(
        self, resulting_synapse: base_models.BaseSynapse, initial_synapse: bt.Synapse
    ) -> Optional[BaseModel]:
        if resulting_synapse and resulting_synapse.dendrite.status_code == 200 and resulting_synapse != initial_synapse:
            formatted_response = self._extract_response(resulting_synapse, initial_synapse)

            return formatted_response
        else:
            return None

    def _extract_response(
        self, resulting_synapse: base_models.BaseSynapse, outgoing_model: BaseModel
    ) -> Optional[BaseModel]:
        try:
            formatted_response = outgoing_model(**resulting_synapse.dict())
            # deserialized_result = resulting_synapse.deserialize()
            # if deserialized_result is None:
            #     formatted_response = None
            return formatted_response
        except ValidationError as e:
            bt.logging.debug(f"FAiled to deserialize for some reason: {e}")
            return None

    def set_weights(self):
        bt.logging.info("Setting weights!")

        uid_scores: Dict[int, List[float]] = {}
        scoring_periods_uid_was_in: Dict[int, int] = {}

        for epoch in self.previous_uid_infos:
            for uid_info in epoch:
                if len(uid_info.available_tasks) == 0:
                    continue

                scoring_periods_uid_was_in[uid_info.uid] = scoring_periods_uid_was_in.get(uid_info.uid, 0) + 1
                if uid_info.request_count == 0:
                    continue

                average_score = uid_info.total_score / max(uid_info.request_count, 1)
                available_tasks = uid_info.available_tasks

                multiplier = cst.AVAILABLE_TASKS_MULTIPLIER[len(available_tasks)]
                score = (multiplier * average_score) ** 2

                uid_scores[uid_info.uid] = uid_scores.get(uid_info.uid, []) + [score]

        uid_weights: Dict[int, float] = {}
        max_periods = max([i for i in scoring_periods_uid_was_in.values()])
        if max_periods == 0:
            bt.logging.info("No uids found to score, nothing to set")
            return
        for uid, periods_for_uid in scoring_periods_uid_was_in.items():
            scores = uid_scores.get(uid, [cst.FAILED_RESPONSE_SCORE])
            average_score = sum(scores) / len(scores)

            uid_weights[uid] = average_score * (periods_for_uid / max_periods) ** 0.5

        weights_tensor = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        for uid, weight in uid_weights.items():
            weights_tensor[uid] = weight

        try:
            netuid = self.config.netuid
            if netuid is None:
                netuid = 19
        except AttributeError:
            netuid = 19

        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=weights_tensor.to("cpu"),
            netuid=netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        NUM_TIMES_TO_SET_WEIGHTS = 3
        # The reason we do this is because wait_for_inclusion & wait_for_finalization
        # Cause the whole API server to crash.
        # So we have no choice but to set weights
        bt.logging.info(f"\n\nSetting weights {NUM_TIMES_TO_SET_WEIGHTS} times without inclusion or finalization\n\n")
        for i in range(NUM_TIMES_TO_SET_WEIGHTS):
            bt.logging.info(f"Setting weights, iteration number: {i+1}")
            success = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=netuid,
                uids=processed_weight_uids,
                weights=processed_weights,
                version_key=VERSION_KEY,
                wait_for_finalization=False,
                wait_for_inclusion=False,
            )

            if success:
                bt.logging.info("✅ Done setting weights!")
            time.sleep(30)


core_validator = CoreValidator()
