import sqlite3
import json
from typing import List, Dict, Any, Optional
from core import constants as core_cst

import bittensor as bt


class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(core_cst.VALIDATOR_DB)
        self._create_tasks_table()

    def _create_tasks_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_name TEXT,
                checking_data TEXT
            )
        """
        )

    def get_tasks_and_number_of_results(self) -> Dict[str, int]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT task_name, COUNT(*) FROM tasks GROUP BY task_name")
        rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    def insert_task_results(
        self, task: str, result: Dict[str, Any], synapse: bt.Synapse, synthetic_query: bool
    ) -> None:
        cursor = self.conn.cursor()
        data_to_store = {
            "result": json.dumps(result),
            "synapse": json.dumps(synapse.dict()),
            "synthetic_query": synthetic_query,
        }
        data = json.dumps(data_to_store)
        cursor.execute("INSERT INTO tasks (task_name, checking_data) VALUES (?,?)", (task, data))
        self.conn.commit()

    def select_and_delete_task_result(self, task: str) -> Optional[List[Dict[str, Any]]]:
        cursor = self.conn.cursor()

        # Query to fetch one result
        cursor.execute("SELECT checking_data FROM tasks WHERE task_name = ? LIMIT 1", (task,))
        row = cursor.fetchone()
        if row is None:
            return None

        result = json.loads(row[0])

        cursor.execute("DELETE FROM tasks WHERE task_name = ? AND checking_data = ?", (task, row[0]))
        self.conn.commit()

        return result

    def close(self):
        self.conn.close()
