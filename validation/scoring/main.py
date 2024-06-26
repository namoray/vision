# Schema for the db
import asyncio
import random

import bittensor as bt
import substrateinterface
from core import Task
from validation.models import RewardData
from validation.proxy.utils import constants as cst
import httpx
from config.validator_config import config as validator_config
from models import utility_models
from validation.proxy import validation_utils, work_and_speed_functions
import json
from validation.db.db_management import db_manager
from validation.db import post_stats
import os
import binascii


def _generate_uid() -> str:
    random_blob = os.urandom(16)
    uid = binascii.hexlify(random_blob).decode("utf-8")
    return uid


def _get_sleep_time(consecutive_errors: float) -> float:
    sleep_time = 0
    if consecutive_errors == 1:
        sleep_time = 60 * 5
    elif consecutive_errors == 2:
        sleep_time = 60 * 10
    elif consecutive_errors == 3:
        sleep_time = 60 * 15
    elif consecutive_errors >= 4:
        sleep_time = 60 * 20

    bt.logging.error(f"Sleeping for {sleep_time} seconds after a http error with the orchestrator server")
    return sleep_time


class Scorer:
    def __init__(self, validator_hotkey: str, testnet: bool, keypair: substrateinterface.Keypair) -> None:
        self.am_scoring_results = False
        self.validator_hotkey = validator_hotkey
        self.testnet = testnet
        self.keypair = keypair

    def start_scoring_results_if_not_already(self):
        if not self.am_scoring_results:
            asyncio.create_task(self._score_results())

        self.am_scoring_results = True

    async def _score_results(self):
        while True:
            tasks_and_number_of_results = db_manager.get_tasks_and_number_of_results()
            total_tasks_stored = sum(tasks_and_number_of_results.values())

            if total_tasks_stored < cst.MINIMUM_TASKS_TO_START_SCORING:
                await asyncio.sleep(5)
                continue

            else:
                task_to_score = random.choices(
                    list(tasks_and_number_of_results.keys()), weights=list(tasks_and_number_of_results.values()), k=1
                )[0]

                await self._check_scores_for_task(Task(task_to_score))
                await asyncio.sleep(5)

    async def _check_scores_for_task(self, task: Task) -> None:
        i = 0
        consecutive_errors = 0
        bt.logging.info(f"Checking some results for task {task}")
        while i < cst.MAX_RESULTS_TO_SCORE_FOR_TASK:
            data_and_hotkey = db_manager.select_and_delete_task_result(task)  # noqa
            if data_and_hotkey is None:
                bt.logging.error(f"No data to score for task {task}")
                return
            checking_data, miner_hotkey = data_and_hotkey
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
                "task": task.value,
            }
            async with httpx.AsyncClient(timeout=180) as client:
                bt.logging.info(f"\nPosting result {i} for task {task} to be scored\n")
                try:
                    response = await client.post(
                        validator_config.external_server_url + "check-result",
                        data=json.dumps(data),
                    )
                    response.raise_for_status()

                except httpx.HTTPStatusError as stat_err:
                    bt.logging.error(f"When scoring, HTTP status error occurred: {stat_err}")
                    consecutive_errors += 1
                    await asyncio.sleep(_get_sleep_time(consecutive_errors))
                    continue

                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout) as read_err:
                    bt.logging.error(f"When scoring, Read timeout occurred: {read_err}")
                    consecutive_errors += 1
                    await asyncio.sleep(_get_sleep_time(consecutive_errors))
                    continue

                except httpx.HTTPError as http_err:
                    consecutive_errors += 1
                    bt.logging.error(f"When scoring, HTTP error occurred: {http_err}")
                    if response.status_code == 502:
                        bt.logging.error("Is your orchestrator server running?")
                    else:
                        bt.logging.error(f"Status code: {response.status_code}")
                    await asyncio.sleep(_get_sleep_time(consecutive_errors + 1))
                    continue

            consecutive_errors = 0
            try:
                response_json = response.json()
                axon_scores = response_json["axon_scores"]
            except (json.JSONDecodeError, KeyError) as parse_err:
                bt.logging.error(f"Error occurred when parsing the response: {parse_err}")
                continue

            score_with_old_speed = await validation_utils.get_expected_score(
                utility_models.QueryResult(**results_json), synapse, task
            )
            volume = work_and_speed_functions.calculate_work(task=task, result=results_json, synapse=synapse)
            speed_scoring_factor = work_and_speed_functions.calculate_speed_modifier(
                task=task, result=results_json, synapse=synapse
            )
            for uid, score in axon_scores.items():
                # We divide max_expected_score whilst the orchestrator is still factoring this into the score
                # once it's removed from orchestrator, we'll remove it from here
                if score == 0 or score_with_old_speed == 0:
                    quality_score = 0
                else:
                    quality_score = score / score_with_old_speed

                id = _generate_uid()

                reward_data = RewardData(
                    id=id,
                    task=task.value,
                    axon_uid=int(uid),
                    quality_score=quality_score,
                    validator_hotkey=self.validator_hotkey,  # fix
                    miner_hotkey=miner_hotkey,
                    synthetic_query=synthetic_query,
                    response_time=results_json["response_time"] if score != 0 else None,
                    volume=volume,
                    speed_scoring_factor=speed_scoring_factor,
                )
                uid = db_manager.insert_reward_data(reward_data)

                data_to_post = reward_data.dict()
                data_to_post[cst.TESTNET] = self.testnet

                await post_stats.post_to_tauvision(
                    data_to_post=data_to_post,
                    keypair=self.keypair,
                    data_type_to_post=post_stats.DataTypeToPost.REWARD_DATA,
                )

            i += 1
