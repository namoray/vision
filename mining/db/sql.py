from mining.db import constants as cst


def select_tasks_and_number_of_results() -> str:
    return f"""
        SELECT {cst.TASK_NAME}, {cst.VOLUME}, {cst.CONCURRENCY_GROUP_ID}
        FROM {cst.TASKS_CONFIG_TABLE}
        WHERE {cst.MINER_HOTKEY} = ?;
    """


def insert_default_task_configs() -> str:
    return f"""
        INSERT INTO {cst.TASKS_CONFIG_TABLE} ({cst.TASK_NAME}, {cst.VOLUME}, {cst.CONCURRENCY_GROUP_ID}, {cst.MINER_HOTKEY}) VALUES (?, ?, ?, ?);
    """


def insert_default_task_concurrency_group_configs() -> str:
    return f"""
        INSERT INTO {cst.TASKS_CONCURRENCY_CONFIG_TABLE} ({cst.CONCURRENCY_GROUP_ID}, {cst.CONCURRENCY_GROUP_LIMIT}) VALUES (?, ?);
    """


def search_concurrency_group_config() -> str:
    return f"""
        SELECT 1 FROM {cst.TASKS_CONCURRENCY_CONFIG_TABLE}
        WHERE {cst.CONCURRENCY_GROUP_ID} = ?;
    """


def search_task_config() -> str:
    return f"""
        SELECT 1 FROM {cst.TASKS_CONFIG_TABLE}
        WHERE {cst.TASK_NAME} = ? AND {cst.MINER_HOTKEY} = ?;
    """


def load_concurrency_groups() -> str:
    return f"""
        SELECT {cst.CONCURRENCY_GROUP_ID}, {cst.CONCURRENCY_GROUP_LIMIT}
        FROM {cst.TASKS_CONCURRENCY_CONFIG_TABLE};
    """


def load_task_capacities() -> str:
    return f"""
        SELECT {cst.TASK_NAME}, {cst.VOLUME}, {cst.CONCURRENCY_GROUP_ID}
        FROM {cst.TASKS_CONFIG_TABLE}
        WHERE {cst.MINER_HOTKEY} = ?;
    """
