from datetime import datetime, timedelta
import random
import sqlite3
import json
from typing import List, Dict, Any, Optional, Union
from core import Task, constants as core_cst

import bittensor as bt

from models import utility_models
from validation.db import sql
from validation.models import PeriodScore, RewardData, UIDRecord

MAX_TASKS_IN_DB_STORE = 1000


class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(core_cst.VISION_DB)
        self.task_weights: Dict[Task, float] = {}

    def get_tasks_and_number_of_results(self) -> Dict[str, int]:
        cursor = self.conn.cursor()
        cursor.execute(sql.select_tasks_and_number_of_results())
        rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    def _get_number_of_these_tasks_already_stored(self, task: Task) -> int:
        cursor = self.conn.cursor()
        cursor.execute(sql.select_count_rows_of_task_stored_for_scoring(), (task.value,))
        row_count = cursor.fetchone()[0]
        return row_count

    def potentially_store_result_in_sql_lite_db(
        self, result: utility_models.QueryResult, task: Task, synapse: bt.Synapse, synthetic_query: bool
    ) -> None:
        if task not in self.task_weights:
            bt.logging.error(f"{task} not in task weights in db_manager")
            return
        target_percentage = self.task_weights[task]
        target_number_of_tasks_to_store = int(MAX_TASKS_IN_DB_STORE * target_percentage)

        number_of_these_tasks_already_stored = db_manager._get_number_of_these_tasks_already_stored(task)
        if number_of_these_tasks_already_stored <= target_number_of_tasks_to_store:
            db_manager.insert_task_results(task.value, result, synapse, synthetic_query)
        else:
            actual_percentage = number_of_these_tasks_already_stored / MAX_TASKS_IN_DB_STORE
            probability_to_score_again = ((target_percentage / actual_percentage - target_percentage) ** 4)
            if random.random() < probability_to_score_again:
                db_manager.insert_task_results(task.value, result, synapse, synthetic_query)

    def insert_task_results(
        self, task: str, result: utility_models.QueryResult, synapse: bt.Synapse, synthetic_query: bool
    ) -> None:
        cursor = self.conn.cursor()

        cursor.execute(sql.select_count_of_rows_in_tasks())
        row_count = cursor.fetchone()[0]

        if row_count >= MAX_TASKS_IN_DB_STORE + 10:
            cursor.execute(sql.delete_oldest_rows_from_tasks(limit=10))

        data_to_store = {
            "result": result.json(),
            "synapse": json.dumps(synapse.dict()),
            "synthetic_query": synthetic_query,
        }
        hotkey = result.miner_hotkey
        data = json.dumps(data_to_store)
        cursor.execute(sql.insert_task(), (task, data, hotkey))
        self.conn.commit()

    def select_and_delete_task_result(self, task: Task) -> Optional[Union[List[Dict[str, Any]], str]]:
        cursor = self.conn.cursor()

        cursor.execute(sql.select_task_for_deletion(), (task.value,))
        row = cursor.fetchone()
        if row is None:
            return None

        checking_data, miner_hotkey = row
        checking_data_loaded = json.loads(checking_data)

        cursor.execute(sql.delete_specific_task(), (task.value, checking_data))
        self.conn.commit()

        return checking_data_loaded, miner_hotkey

    def insert_reward_data(
        self,
        reward_data: RewardData,
    ) -> str:
        cursor = self.conn.cursor()

        cursor.execute(
            sql.insert_reward_data(),
            (
                reward_data.id,
                reward_data.task,
                reward_data.axon_uid,
                reward_data.quality_score,
                reward_data.validator_hotkey,
                reward_data.miner_hotkey,
                reward_data.synthetic_query,
                reward_data.speed_scoring_factor,
                reward_data.response_time,
                reward_data.volume,
            ),
        )
        self.conn.commit()
        return id

    def clean_tables_of_hotkeys(self, miner_hotkeys: List[str]) -> None:
        cursor = self.conn.cursor()

        for hotkey in miner_hotkeys:
            cursor.execute(sql.delete_task_by_hotkey(), (hotkey,))
            cursor.execute(sql.delete_reward_data_by_hotkey(), (hotkey,))
            cursor.execute(sql.delete_uid_data_by_hotkey(), (hotkey,))
        self.conn.commit()

    def delete_tasks_older_than_date(self, minutes: int) -> None:
        cursor = self.conn.cursor()

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        cutoff_time_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(sql.delete_task_data_older_than(), (cutoff_time_str,))
        self.conn.commit()

    def delete_data_older_than_date(self, minutes: int) -> None:
        cursor = self.conn.cursor()

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        cutoff_time_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(sql.delete_reward_data_older_than(), (cutoff_time_str,))
        cursor.execute(sql.delete_uid_data_older_than(), (cutoff_time_str,))
        cursor.execute(sql.delete_task_data_older_than(), (cutoff_time_str,))

        self.conn.commit()

    def fetch_recent_most_rewards_for_uid(
        self, task: Task, miner_hotkey: str, quality_tasks_to_fetch: int = 30
    ) -> List[RewardData]:
        cursor = self.conn.cursor()
        now = datetime.now()
        cut_off = now - timedelta(hours=72)
        cut_off_timestamp = cut_off.timestamp()

        cursor.execute(
            sql.select_recent_reward_data_for_a_task(),
            (task.value, cut_off_timestamp, miner_hotkey),
        )
        priority_results = cursor.fetchall()
        y = len(priority_results)
        cursor.execute(
            sql.select_recent_reward_data(),
            (task.value, cut_off_timestamp, miner_hotkey, quality_tasks_to_fetch - y),
        )
        fill_results = cursor.fetchall()
        reward_data_list = [
            RewardData(
                id=row[0],
                task=row[1],
                axon_uid=row[2],
                quality_score=row[3],
                validator_hotkey=row[4],
                miner_hotkey=row[5],
                synthetic_query=row[6],
                speed_scoring_factor=row[7],
                response_time=row[8],
                volume=row[9],
                created_at=row[10],
            )
            for row in priority_results + fill_results
        ]

        return reward_data_list

    def insert_uid_record(
        self,
        uid_record: UIDRecord,
        validator_hotkey: str,
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            sql.insert_uid_record(),
            (
                uid_record.axon_uid,
                uid_record.hotkey,
                validator_hotkey,
                uid_record.task.value,
                uid_record.declared_volume,
                uid_record.consumed_volume,
                uid_record.total_requests_made,
                uid_record.requests_429,
                uid_record.requests_500,
                uid_record.period_score,
            ),
        )
        self.conn.commit()

    def fetch_hotkey_scores_for_task(
        self,
        task: Task,
        miner_hotkey: str,
    ) -> List[PeriodScore]:
        cursor = self.conn.cursor()
        cursor.execute(sql.select_uid_period_scores_for_task(), (task.value, miner_hotkey))
        rows = cursor.fetchall()

        period_scores = [
            PeriodScore(
                hotkey=miner_hotkey,
                period_score=row[0],
                consumed_volume=row[1],
                created_at=row[2],
            )
            for row in rows
        ]

        return sorted(period_scores, key=lambda x: x.created_at, reverse=True)

    def close(self):
        self.conn.close()


db_manager = DatabaseManager()
