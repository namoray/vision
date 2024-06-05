import asyncio
import re
import threading
from collections import defaultdict, deque
from typing import Dict, Tuple
from typing import List, Any
from typing import Optional
from typing import Set

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core import Task
from core.tasks import TASK_TO_MAX_CAPACITY
import bittensor as bt
from validation.proxy.utils import query_utils
from core import bittensor_overrides as bto
from config import configuration
from config.validator_config import config as validator_config
from models import base_models, synapses, utility_models
from validation.proxy import validation_utils
import json

from validation.scoring.main import Scorer
from validation.uid_manager import UidManager
from validation.weight_setting.main import WeightSetter
from validation.db import post_stats
from validation.db.db_management import db_manager

PROXY_VERSION = "4.0"
# Change this to not be hardcoded, once the orchestrator supports is
ORCHESTRATOR_VERSION = "0.1.0"

_PASCAL_SEP_REGEXP = re.compile("(.)([A-Z][a-z]+)")
_UPPER_FOLLOWING_REGEXP = re.compile("([a-z0-9])([A-Z])")
MAX_PERIODS_TO_LOOK_FOR_SCORE = 16


def _pascal_to_kebab(input_string: str) -> str:
    hyphen_separated = _PASCAL_SEP_REGEXP.sub(r"\1-\2", input_string)
    return _UPPER_FOLLOWING_REGEXP.sub(r"\1-\2", hyphen_separated).lower()


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
        self.metagraph: bt.metagraph = self.subtensor.metagraph(netuid=self.config.netuid, lite=True)
        self.netuid: int = self.config.netuid if self.config.netuid is not None else 19
        self.task_importances = self._get_task_importances()

        is_testnet = validator_config.subtensor_network == "test" or self.netuid != 19

        self.public_hotkey_address = self.keypair.ss58_address

        validation_utils.connect_to_external_server()

        # Make the above class variables instead

        bt.logging(debug=True)

        self.uids: list[int] = []
        self.axon_indexes: list[int] = []
        self.incentives: list[float] = []

        self.uid_to_uid_info: Dict[int, utility_models.UIDinfo] = {}
        self.previous_uid_infos: deque[List[utility_models.UIDinfo]] = deque([], maxlen=MAX_PERIODS_TO_LOOK_FOR_SCORE)

        self.low_incentive_uids: Set[int] = set()

        self.capacities_for_tasks: Dict[Task, Dict[int, float]] = defaultdict(lambda: {})  # task -> uid -> capacity

        self.threading_lock = threading.Lock()

        self.task_id_to_work: Dict[str, float] = {}

        self.results_store: Dict[str, utility_models.QueryResult] = {}

        self.scorer = Scorer(validator_hotkey=self.keypair.ss58_address, testnet=is_testnet, keypair=self.keypair)
        self.weight_setter = WeightSetter(subtensor=self.subtensor, config=self.config)
        self.uid_manager = None

    def _get_task_importances(self) -> Dict[Task, float]:
        """
        TODO: Replace with onchain commitments. For initial testnet release,
        Hardcode to a couple of values
        """
        return {Task.proteus_text_to_image: 0.5, Task.playground_text_to_image: 0.1, Task.chat_llama_3: 0.4}

    def _correct_capacities(self) -> None:
        self._correct_for_max_capacities()
        self._correct_capacities_for_my_stake()

    def _correct_capacities_for_my_stake(self) -> None:
        # TODO: Replace with real stake finding on mainnet - but for testnet, the stakes dont make sense, so will fix it for now
        my_proportion_of_stake = 0.25

        for task in Task:
            capacities = self.capacities_for_tasks[task]
            for uid, capacities_for_uid in capacities.items():
                capacities[uid] = capacities_for_uid * my_proportion_of_stake

    def _correct_for_max_capacities(self) -> None:
        for task in Task:
            capacities = self.capacities_for_tasks[task]
            max_capacity = TASK_TO_MAX_CAPACITY[task]
            for uid, capacity in capacities.items():
                self.capacities_for_tasks[task][uid] = min(capacity, max_capacity)

    def prepare_config_and_logging(self) -> bt.config:
        base_config = configuration.get_validator_cli_config()

        bt.logging(config=base_config, logging_dir=base_config.full_path)
        return base_config

    def start_continuous_tasks(self):
        self.score_task = asyncio.create_task(self.run_vali())
        self.score_task.add_done_callback(validation_utils.log_task_exception)

    async def fetch_available_capacities_for_each_axon(self) -> None:
        uid_to_query_task = {}

        for uid in self.uid_to_uid_info.keys():
            task = asyncio.create_task(
                query_utils.query_individual_axon(
                    synapse=synapses.Capacity(),
                    dendrite=self.dendrite,
                    axon=self.uid_to_uid_info[uid].axon,
                    axon_uid=uid,
                    deserialize=True,
                    log_requests_and_responses=False,
                )
            )

            uid_to_query_task[uid] = task

        responses_and_response_times: List[
            Tuple[Optional[Dict[Task, base_models.VolumeForTask]], float]
        ] = await asyncio.gather(*uid_to_query_task.values())

        uids = uid_to_query_task.keys()
        all_capacities = [i[0] for i in responses_and_response_times]

        bt.logging.info(f"Got capacities from {len([i for i in all_capacities if i is not None])} axons!")
        for uid, capacities in zip(uids, all_capacities):
            if capacities is None:
                continue

            allowed_tasks = set([task for task in Task])
            for task, volume in capacities.items():
                # This is to stop people claiming tasks that don't exist
                if task not in allowed_tasks:
                    continue
                if uid not in self.capacities_for_tasks[task]:
                    self.capacities_for_tasks[task][uid] = volume.volume
        self._correct_capacities()
        capacities_to_log = {k.value: v for k, v in self.capacities_for_tasks.items()}
        bt.logging.info(f"Capacities: {capacities_to_log}")
        bt.logging.info("Done fetching available tasks!")

    async def resync_metagraph(self):
        """
        Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph.
        This really needs to work in a separate runtime environment, or can query an api directly as a first try
        """
        bt.logging.info("Resyncing the metagraph!")
        await asyncio.to_thread(self.metagraph.sync, subtensor=self.subtensor, lite=True)

        bt.logging.info("Got the capacities, now storing the info....")
        incentives_tensor, axon_indexes_tensor = self.metagraph.incentive.sort(descending=True)

        with self.threading_lock:
            self.uid_to_uid_info = {}
            self.uids: List[int] = self.metagraph.uids.tolist()
            self.axon_indexes = axon_indexes_tensor.tolist()
            self.incentives = incentives_tensor.tolist()
            hotkeys: List[str] = self.metagraph.hotkeys  # noqa: F841
            axons = self.metagraph.axons

            for i in self.axon_indexes:
                uid = self.uids[i]
                self.uid_to_uid_info[uid] = utility_models.UIDinfo(
                    uid=uid,
                    axon=axons[i],
                    hotkey=hotkeys[i],
                )

        bt.logging.info("Finished extraction - now to fetch the available capacities for each axon")
        await self.fetch_available_capacities_for_each_axon()

        return

    async def run_vali(self) -> None:
        await post_stats.post_to_tauvision(
            data_to_post={
                "versions": PROXY_VERSION + ":" + ORCHESTRATOR_VERSION,
                "validator_hotkey": self.public_hotkey_address,
            },
            keypair=self.keypair,
            data_type_to_post=post_stats.DataTypeToPost.VALIDATOR_INFO,
        )

        db_manager.delete_data_older_than_date(minutes=72 * 60)
        db_manager.delete_tasks_older_than_date(minutes=120)

        iteration = 1
        while True:
            # Wait for initial syncing of metagraph
            await self.resync_metagraph()
            self.scorer.start_scoring_results_if_not_already()

            bt.logging.info("🚀 Starting to score stuff, metagraph is synced...")
            self.uid_manager = UidManager(
                self.capacities_for_tasks,
                dendrite=self.dendrite,
                uid_to_uid_info=self.uid_to_uid_info,
                validator_hotkey=self.keypair.ss58_address,
            )
            await self.uid_manager.start_synthetic_scoring()

            await self.uid_manager.collect_synthetic_scoring_results()
            self.uid_manager.calculate_period_scores_for_uids()
            self.uid_manager.store_period_scores()

            bt.logging.info(f"Finished scoring for iteration: {iteration}. Now settings weights")
            iteration += 1

            await self.weight_setter.start_weight_setting_process(
                self.metagraph,
                self.wallet,
                self.config.netuid,
                self.capacities_for_tasks,
                self.uid_to_uid_info,
                self.task_importances,
            )

            db_manager.delete_tasks_older_than_date(minutes=120)

            await asyncio.sleep(60)

    async def make_organic_query(
        self, task: Task, stream: bool, outgoing_model: BaseModel, synapse: bt.Synapse
    ) -> JSONResponse:
        if self.uid_manager is None:
            return JSONResponse(status_code=500, content={"message": "Server booting, one sec"})

        return await self.uid_manager.make_orgnanic_query(
            task=task, synapse=synapse, stream=stream, outgoing_model=outgoing_model
        )


core_validator = CoreValidator()
