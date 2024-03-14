import asyncio
import copy
import random
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Callable

import bittensor as bt
import httpx
import torch
from pydantic import BaseModel
from pydantic import ValidationError
import yaml
from core import bittensor_overrides as bto, configuration
from core import constants as core_cst, utils as core_utils
from models import base_models
from models import protocols
from models import utility_models
from validation.validator_api_server import constants as cst
from validation.validator_api_server import validation_utils
from validation.validator_api_server import similarity_comparisons
import json
import traceback


_PASCAL_SEP_REGEXP = re.compile("(.)([A-Z][a-z]+)")
_UPPER_FOLLOWING_REGEXP = re.compile("([a-z0-9])([A-Z])")

VERSION_KEY = 20_002



def _pascal_to_kebab(input_string: str) -> str:
    hyphen_separated = _PASCAL_SEP_REGEXP.sub(r"\1-\2", input_string)
    return _UPPER_FOLLOWING_REGEXP.sub(r"\1-\2", hyphen_separated).lower()


MAX_PERIODS_TO_LOOK_FOR_SCORE = 30


class CoreValidator:
    def __init__(self) -> None:
        self.config = self.prepare_config_and_logging()
        self.subtensor = bt.subtensor(config=self.config)
        self.wallet = bt.wallet(config=self.config)
        self.dendrite = bto.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid, lite=True)

        (
            self.BASE_CHECKING_SERVER_URL,
            self.BASE_SAFETY_CHECKER_SERVER_URL,
        ) = validation_utils.connect_to_checking_servers(self.config)

        # Make the above class variables instead

        bt.logging(debug=True)

        self.uids: list[int] = []
        self.axon_indexes: list[int] = []
        self.incentives: list[float] = []
        self.axons: list[bt.axon] = []

        self.uid_to_uid_info: Dict[int, utility_models.UIDinfo] = {}
        self.previous_uid_infos: deque[List[utility_models.UIDinfo]] = deque([], maxlen=MAX_PERIODS_TO_LOOK_FOR_SCORE)

        self.low_incentive_uids: Set[int] = set()

        self.operations_to_available_axon_uids: Dict[str, Set[int]] = defaultdict(lambda: set())
        self.failed_queries_per_miner: Dict[int, int] = defaultdict(lambda: 0)

        self.threading_lock = threading.Lock()

        self.results_store: Dict[str, utility_models.QueryResult] = {}

    def prepare_config_and_logging(self) -> bt.config:
        yaml_config = yaml.safe_load(open(core_cst.CONFIG_FILEPATH))
        validator_hotkey_name = core_utils.get_validator_hotkey_name_from_config(yaml_config)
        base_config = configuration.get_validator_cli_config(
            yaml_config.get(validator_hotkey_name, {}), validator_hotkey_name
        )

        bt.logging(config=base_config, logging_dir=base_config.full_path)
        return base_config

    def start_continuous_tasks(self):
        self.resync_task = asyncio.create_task(self.periodically_resync_and_set_weights())
        self.resync_task.add_done_callback(validation_utils.log_task_exception)

        self.score_task = asyncio.create_task(self.synthetically_score())
        self.score_task.add_done_callback(validation_utils.log_task_exception)

    async def periodically_resync_and_set_weights(self) -> None:
        time_between_resyncing =  10 * 60
        while True:
            await self.resync_metagraph()
            await asyncio.sleep(time_between_resyncing)
            
            await self.resync_metagraph()
            await asyncio.to_thread(self.set_weights)

            await asyncio.sleep(time_between_resyncing)


    async def _query_checking_server_for_expected_result(
        self, endpoint: str, synapse: bt.Synapse, outgoing_model: BaseModel
    ) -> Optional[utility_models.QueryResult]:
        url = self.BASE_CHECKING_SERVER_URL + core_cst.CHECKING_ENDPOINT_PREFIX + "/" + endpoint

        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(url, data=json.dumps(synapse.dict()))

            if response.status_code == 200:
                formatted_response = outgoing_model(**response.json())
                return utility_models.QueryResult(formatted_response=formatted_response)
        except httpx.HTTPError as e:
            bt.logging.debug(f"Error when querying endpoint {url}: {e}")
            return None

    async def _query_checking_server_for_synthetic_data(self, operation: str) -> Optional[utility_models.QueryResult]:
        endpoint = _pascal_to_kebab(operation)
        url = self.BASE_CHECKING_SERVER_URL + core_cst.SYNTHETIC_ENDPOINT_PREFIX + "/" + endpoint
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.get(url)

            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError as e:
            error_details = str(e)
            traceback_info = traceback.format_exc()
            bt.logging.debug(f"Error when querying endpoint {url}: {error_details}\n Traceback information: {traceback_info}")
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

        while True:
            operation = random.choice(cst.OPERATIONS_TO_SCORE_SYNTHETICALLY)
            synthetic_data = await self._query_checking_server_for_synthetic_data(operation)

            if synthetic_data is None:
                bt.logging.error(
                    f"Synthetic data is none which is weird, will try again in {cst.MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING}."
                    "Maybe the server hasn't finished initialising yet"
                )
                await asyncio.sleep(cst.MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING)
                continue

            synapse_class_ = getattr(protocols, operation)
            synapse = synapse_class_(**synthetic_data)
            outgoing_model = getattr(base_models, operation + core_cst.OUTGOING)

            time_before_query = time.time()

            await self.execute_query(synapse, outgoing_model, synthetic_query=True)

            time_to_execute_query = time.time() - time_before_query
            await asyncio.sleep(max(cst.MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING - time_to_execute_query, 0))

    async def fetch_available_operations_for_each_axon(self) -> None:
        uid_to_query_task = {}

        for uid in self.uid_to_uid_info.keys():
            task = asyncio.create_task(
                self.query_individual_axon(
                    synapse=protocols.AvailableOperations(),
                    axon_uid=uid,
                    deserialize=True,
                    log_requests_and_responses=True,
                )
            )
            uid_to_query_task[uid] = task

        responses_and_response_times: List[Tuple[Optional[Dict[str, bool]], float]] = await asyncio.gather(
            *uid_to_query_task.values()
        )

        uids = uid_to_query_task.keys()
        all_available_operations = [i[0] for i in responses_and_response_times]

        for uid, available_operations in zip(uids, all_available_operations):
            if available_operations is None:
                continue

            operations_for_uid = []

            for operation_name, available_flag in available_operations.items():
                if operation_name not in cst.ALLOWED_USEFUL_OPERATIONS:
                    continue
                if available_flag:
                    self.operations_to_available_axon_uids[operation_name].add(uid)
                operations_for_uid.append(operation_name)

            self.uid_to_uid_info[uid].available_operations = operations_for_uid
            bt.logging.debug(f"Available operations for uid {uid}: {operations_for_uid}")

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
                    uid=uid, axon=self.axons[i], incentive=self.incentives[i], hotkey=hotkeys[i]
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
        await self.fetch_available_operations_for_each_axon()

        return

    def _get_similarity_comparison_function(self, outgoing_model: str) -> Callable:
        return similarity_comparisons.SYNAPSE_TO_COMPARISON_FUNCTION[outgoing_model]

    async def query_individual_axon(
        self, synapse: bt.Synapse, axon_uid: int, deserialize: bool = False, log_requests_and_responses: bool = True
    ) -> Tuple[base_models.BaseSynapse, float]:
        operation_name = synapse.__class__.__name__
        if operation_name not in cst.OPERATION_TIMEOUTS:
            bt.logging.warning(
                f"Operation {operation_name} not in operation_to_timeout, this is probably a mistake / bug 🐞"
            )
        
        start_time = time.time()

        response = await self.dendrite.forward(
            axons=self.uid_to_uid_info[axon_uid].axon,
            synapse=synapse,
            connect_timeout=2,
            response_timeout=cst.OPERATION_TIMEOUTS.get(operation_name, 15),
            deserialize=deserialize,
            log_requests_and_responses=log_requests_and_responses,
            streaming=False
        )
        return response, time.time() - start_time

    async def execute_query(
        self, synapse: bt.Synapse, outgoing_model: BaseModel, synthetic_query: bool = False
    ) -> utility_models.QueryResult:
        operation_name = synapse.__class__.__name__

        available_axons = self._get_available_axons(operation_name)
        if not available_axons:
            bt.logging.warning(f"No axons available for query for {operation_name}")
            return utility_models.QueryResult(error_message=f"No axons available for operation {operation_name}")

        should_score = synthetic_query or random.random() < cst.SCORE_QUERY_PROBABILITY
        sufficient_secondary_axons = len(available_axons) > cst.NUMBER_OF_SECONDARY_AXONS_TO_COMPARE_WHEN_SCORING

        if should_score and sufficient_secondary_axons:
            bt.logging.info("Scoring a query!")

            axons_to_comparitively_score = random.sample(
                available_axons, cst.NUMBER_OF_SECONDARY_AXONS_TO_COMPARE_WHEN_SCORING
            )

            task_id_of_main_response = str(uuid.uuid4())
            task_ids_of_secondary_responses = [str(uuid.uuid4()) for _ in axons_to_comparitively_score]

            task_ids_to_comparitive_axon_uids = {
                task_ids_of_secondary_responses[i]: axons_to_comparitively_score[i]
                for i in range(len(axons_to_comparitively_score))
            }

            bt.logging.info(f"Querying {axons_to_comparitively_score} to score with the main response!")
            asyncio.create_task(
                self._query_miners_for_comparitive_scores_and_score_them_all(
                    synapse,
                    task_id_of_main_response,
                    task_ids_of_secondary_responses,
                    task_ids_to_comparitive_axon_uids,
                    outgoing_model,
                    synthetic_query,
                )
            )

            miners_to_query_order = self._get_miners_query_order(
                available_axons, axons_to_comparitively_score, synthetic_query
            )
            main_query_result = await self._query_miners_until_result(miners_to_query_order, synapse, outgoing_model)
            self.results_store[task_id_of_main_response] = main_query_result
            return main_query_result

        else:
            miners_to_query_order = self._get_miners_query_order(available_axons)
            main_query_result = await self._query_miners_until_result(miners_to_query_order, synapse, outgoing_model)
            return main_query_result

    async def _query_miners_for_comparitive_scores_and_score_them_all(
        self,
        synapse: bt.Synapse,
        task_id_of_main_response: str,
        task_ids_of_secondary_responses: List[str],
        task_ids_to_axon_uids: Dict[str, int],
        outgoing_model: BaseModel,
        synthetic_query: bool,
    ):
        for task_id in task_ids_of_secondary_responses:
            asyncio.create_task(
                self._query_miner_and_store_result_in_store(
                    synapse, task_ids_to_axon_uids[task_id], outgoing_model, task_id
                )
            )

        all_task_ids = [task_id_of_main_response] + task_ids_of_secondary_responses

        # Each result will be of the type QueryResult when it is finished & stored
        while any(self.results_store.get(task_id) is None for task_id in all_task_ids):
            await asyncio.sleep(2)

        results: List[utility_models.QueryResult] = []
        # Clear the results store for the latest results, and store for later
        for task_id in all_task_ids:
            result = self.results_store.pop(task_id)
            results.append(result)

        # TODO for future improvement: remove & add More generic implementation for > 2 comparisons
        assert len(results) == 2

        result1, result2 = results[0], results[1]

        # If both None, then the query was probably invalid, so there's nothing to score
        if result1.formatted_response is None and result2.formatted_response is None:
            dict_to_log = core_utils.model_to_printable_dict(synapse)
            bt.logging.error(
                f"😱 Just got two none results. Please let the dev know!\n Synapse: {dict_to_log}; is synthetic? {synthetic_query};"
            )
            return



        axon_scores: Dict[int, float] = {}

        for result in results:
            for failed_axon in result.failed_axon_uids:
                if failed_axon is not None:
                    axon_scores[failed_axon] = cst.FAILED_RESPONSE_SCORE
        
        

        similarity_comparison_function = self._get_similarity_comparison_function(synapse.__class__.__name__)
        responses_are_similar = similarity_comparison_function(result1.formatted_response, result2.formatted_response)
        checked_with_server = False

        # If they are similar, only use the external server to check the responses some of the time
        if responses_are_similar and random.random() > cst.CHANCE_TO_CHECK_OUTPUT_WHEN_IMAGES_FROM_MINERS_WERE_SIMILAR:
            bt.logging.info("Checking scores without server...")
            compared_axon_scores = self._get_axon_scores_without_server_check(result1, result2)

        else:
            bt.logging.info("Checking scores with server...")
            compared_axon_scores = await self._get_axon_scores_with_server_check(
                result1, result2, synapse, outgoing_model, similarity_comparison_function
            )
            checked_with_server = True
        
        axon_scores = {**axon_scores, **compared_axon_scores}
        
        validation_utils.store_and_print_scores(axon_scores, result1, result2, synapse, checked_with_server, self.uid_to_uid_info)

        quickest_response_time = min(
            (t for t in [result1.response_time, result2.response_time] if t is not None), default=1
        )
        count = max(int(quickest_response_time), 1)
        for axon_uid, score in axon_scores.items():
            if axon_uid is None:
                bt.logging.error(
                    f"axon_uid is None, score: {score}, results: {core_utils.model_to_printable_dict(result1)}, {core_utils.model_to_printable_dict(result2)}"
                )
                continue
            uid_info = self.uid_to_uid_info[axon_uid]
            uid_info.add_score(score, synthetic=synthetic_query, count=count)

    async def _get_axon_scores_with_server_check(
        self,
        result1: utility_models.QueryResult,
        result2: utility_models.QueryResult,
        synapse: bt.synapse,
        outgoing_model: BaseModel,
        similarity_comparison_function: Callable,
    ) -> Dict[int, float]:
        axon_scores = {}
        endpoint = _pascal_to_kebab(synapse.__class__.__name__)
        expected_result = await self._query_checking_server_for_expected_result(endpoint, synapse, outgoing_model)

        faster_response_bonus = 1 + cst.BONUS_FOR_WINNING_MINER
        slower_response_penalty = 1 - cst.BONUS_FOR_WINNING_MINER

        bt.logging.info("Got expected result")
        # We know that at least one result is not None, so we're not expecting None here.
        # This means if expected result is None, there's a problem with the checking server
        if expected_result is None:
            printable_synapse = core_utils.model_to_printable_dict(synapse)
            printable_outgoing_model = core_utils.model_to_printable_dict(outgoing_model)
            bt.logging.error(f"Could not get expected output from server, which is weird! Please raise this with the subnet devs. Synapse: {printable_synapse}, outgoing_model: {printable_outgoing_model}")
            return {}
        
        # Otherwise, get the respective similarities with the server and then compare response times


        result1_is_similar_to_truth = similarity_comparison_function(
            result1.formatted_response, expected_result.formatted_response
        )
        result2_is_similar_to_truth = similarity_comparison_function(
            result2.formatted_response, expected_result.formatted_response
        )

        if result1_is_similar_to_truth == 1 and result2_is_similar_to_truth == 1:
            if result1.response_time < result2.response_time:
                axon_scores[result1.axon_uid] = faster_response_bonus
                axon_scores[result2.axon_uid] = slower_response_penalty
            else:
                axon_scores[result1.axon_uid] = slower_response_penalty
                axon_scores[result2.axon_uid] = faster_response_bonus
        
        else:

            if not result1_is_similar_to_truth == 1:
                
                if not result2_is_similar_to_truth == 1:
                    axon_scores[result1.axon_uid] = max(result1_is_similar_to_truth, cst.FAILED_RESPONSE_SCORE)
                    axon_scores[result2.axon_uid] = max(result2_is_similar_to_truth, cst.FAILED_RESPONSE_SCORE)
                else:
                    axon_scores[result1.axon_uid] = max(slower_response_penalty * result1_is_similar_to_truth, cst.FAILED_RESPONSE_SCORE)
                    axon_scores[result2.axon_uid] = faster_response_bonus
            else:
                axon_scores[result1.axon_uid] = faster_response_bonus
                axon_scores[result2.axon_uid] = max(slower_response_penalty * result2_is_similar_to_truth, cst.FAILED_RESPONSE_SCORE)

        return axon_scores

    def _get_axon_scores_without_server_check(
        self, result1: utility_models.QueryResult, result2: utility_models.QueryResult
    ) -> Dict[int, float]:
        
        if result1.response_time is None or result2.response_time is None:

            bt.logging.error(
                "Some Response time is none in without server check! Why!?\n"
                f"result 1: {core_utils.model_to_printable_dict(result1)}"
                f"\nresult 2: {core_utils.model_to_printable_dict(result2)}"
            )
            return {}
        
        assert not all([result.formatted_response is None for result in [result1, result2]])

        axon_scores = {}
        faster_response_bonus = 1 + cst.BONUS_FOR_WINNING_MINER
        slower_response_penalty = 1 - cst.BONUS_FOR_WINNING_MINER

        if result1.response_time < result2.response_time:
            axon_scores[result1.axon_uid] = faster_response_bonus
            axon_scores[result2.axon_uid] = slower_response_penalty
        else:
            axon_scores[result2.axon_uid] = faster_response_bonus
            axon_scores[result1.axon_uid] = slower_response_penalty

        return axon_scores

    async def _query_miner_and_store_result_in_store(
        self, synapse: bt.synapse, axon_uid: int, outgoing_model: BaseModel, task_id: str
    ):
        resulting_synapse, response_time = await self.query_individual_axon(synapse, axon_uid)

        formatted_response = self._get_formatted_response(resulting_synapse, outgoing_model)

        result = utility_models.QueryResult(
            formatted_response=formatted_response,
            axon_uid=axon_uid,
            failed_axon_uids=[],
            response_time=response_time if formatted_response is not None else None,
            error_message=resulting_synapse.error_message,
        )
        self.results_store[task_id] = result

    def _get_available_axons(self, operation_name: str) -> List[int]:
        return list(self.operations_to_available_axon_uids.get(operation_name, []))

    def _get_miners_query_order(
        self, available_axons: List[int], axons_to_exclude: List[int] = [], synthetic_query: bool = False
    ) -> list:
        if axons_to_exclude:
            available_axons = [axon for axon in available_axons if axon not in axons_to_exclude]

        if synthetic_query:
            axons_to_number_of_queries = {
                axon: self.uid_to_uid_info[axon].synthetic_request_count for axon in available_axons
            }

        else:
            available_axons = [axon for axon in available_axons if axon not in self.low_incentive_uids]
            axons_to_number_of_queries = {
                axon: self.uid_to_uid_info[axon].organic_request_count for axon in available_axons
            }

        queries_to_axons = defaultdict(list)
        for axon, num_queries in axons_to_number_of_queries.items():
            queries_to_axons[num_queries].append(axon)

        sorted_query_groups = sorted(queries_to_axons.items(), key=lambda item: item[0])

        query_order = []
        for _, axons in sorted_query_groups:
            random.shuffle(axons)
            query_order.extend(axons)

        return query_order

    async def _query_miners_until_result(
        self, miners_to_query_order: list, synapse: bt.Synapse, outgoing_model: BaseModel
    ) -> utility_models.QueryResult:
        internal_server_errors = 0
        failed_axon_uids = []

        for axon_uid in miners_to_query_order:
            bt.logging.debug(f"Querying axon: {axon_uid} for synapse {synapse.__class__.__name__}")
            resulting_synapse, response_time = await self.query_individual_axon(synapse, axon_uid)

            formatted_response = self._get_formatted_response(resulting_synapse, outgoing_model)
            if formatted_response is not None:
                return utility_models.QueryResult(
                    formatted_response=formatted_response,
                    axon_uid=axon_uid,
                    failed_axon_uids=failed_axon_uids,
                    response_time=response_time,
                    error_message=resulting_synapse.error_message,
                )

            bt.logging.debug(f"Failed response from axon: {axon_uid} for query of {synapse.__class__.__name__} :(")

            internal_server_errors += 1
            failed_axon_uids.append(axon_uid)
            if internal_server_errors >= cst.MAX_INTERNAL_SERVER_ERRORS:
                bt.logging.debug(
                    f"Too many internal server errors, something is wrong with your request. Message: {resulting_synapse.error_message}"
                )
                return utility_models.QueryResult(
                    error_message=resulting_synapse.error_message, failed_axon_uids=failed_axon_uids, axon_uid=axon_uid
                )

        return utility_models.QueryResult(
            error_message="Unable to get a valid response from any axon", failed_axon_uids=failed_axon_uids, axon_uid=axon_uid
        )

    def _get_formatted_response(
        self, resulting_synapse: base_models.BaseSynapse, initial_synapse: bt.Synapse
    ) -> Optional[BaseModel]:
        if (
            resulting_synapse
            and resulting_synapse.dendrite.status_code == 200
            and (
                resulting_synapse.error_message is None
                or core_cst.NSFW_RESPONSE_ERROR in resulting_synapse.error_message.lower()
            )
            and resulting_synapse != initial_synapse
        ):
            formatted_response = self._extract_response_and_check_deserialization(resulting_synapse, initial_synapse)

            return formatted_response
        else:
            return None

    def _extract_response_and_check_deserialization(
        self, resulting_synapse: base_models.BaseSynapse, outgoing_model: BaseModel
    ) -> Optional[BaseModel]:
        try:
            formatted_response = outgoing_model(**resulting_synapse.dict())
            deserialized_result = resulting_synapse.deserialize()
            if deserialized_result is None:
                formatted_response = None
            return formatted_response
        except ValidationError as e:
            bt.logging.debug(f"FAiled to deserialize for some reason: {e}")
            return None

    def set_weights(self):
        bt.logging.info("Setting weights!")

        # TODO: CHANGE THIS
        uid_scores: Dict[int, List[float]] = {}
        scoring_periods_uid_was_in: Dict[int, int] = {}

        for epoch in self.previous_uid_infos:
            for uid_info in epoch:

                if len(uid_info.available_operations) == 0:
                    continue
            
                scoring_periods_uid_was_in[uid_info.uid] = scoring_periods_uid_was_in.get(uid_info.uid, 0) + 1
                if uid_info.organic_request_count + uid_info.synthetic_request_count == 0:
                    continue

                average_score = uid_info.average_score
                available_operations = uid_info.available_operations

                multiplier = cst.AVAILABLE_OPERATIONS_MULTIPLIER[len(available_operations)]
                score = multiplier * average_score

                uid_scores[uid_info.uid] = uid_scores.get(uid_info.uid, []) + [score]

        uid_weights: Dict[int, float] = {}
        max_periods = max([i for i in scoring_periods_uid_was_in.values()])
        if max_periods == 0:
            bt.logging.info("No uids found to score, nothing to set")
            return
        for uid, periods_for_uid in scoring_periods_uid_was_in.items():
            scores = uid_scores.get(uid, [0.5])
            average_score = sum(scores) / len(scores)

            uid_weights[uid] = average_score * (periods_for_uid / max_periods) ** 0.5

        if uid_weights == {}:
            bt.logging.info("No scores found, nothing to set")
            return

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
        bt.logging.info(f"Setting weights {NUM_TIMES_TO_SET_WEIGHTS} times without inclusion or finalization")
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
