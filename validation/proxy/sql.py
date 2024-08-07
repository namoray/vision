import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from validation.db.db_management import db_lock

BALANCE = "balance"
KEY = "key"
NAME = "name"
COST = "cost"
ENDPOINT = "endpoint"


RATE_LIMIT_PER_MINUTE = "rate_limit_per_minute"
API_KEYS_TABLE = "api_keys"
LOGS_TABLE = "logs"
CREATED_AT = "created_at"

DATABASE_PATH = "vision_database.db"


async def get_db_connection():
    conn = await aiosqlite.connect(DATABASE_PATH)
    await conn.execute("PRAGMA foreign_keys = ON")
    return conn


async def get_api_key_info(conn: aiosqlite.Connection, api_key: str) -> Optional[Dict[str, Any]]:
    async with db_lock:
        async with conn.execute(
            f"SELECT {KEY}, {BALANCE}, {RATE_LIMIT_PER_MINUTE} FROM {API_KEYS_TABLE} WHERE {KEY} = ?", (api_key,)
        ) as cursor:
            row = await cursor.fetchone()
            return (
                {
                    KEY: row[0],
                    BALANCE: row[1],
                    RATE_LIMIT_PER_MINUTE: row[2],
                }
                if row
                else None
            )


async def get_all_api_keys(conn: aiosqlite.Connection) -> List[Dict[str, Any]]:
    async with db_lock:
        async with conn.execute(f"SELECT {KEY}, {BALANCE}, {RATE_LIMIT_PER_MINUTE} FROM {API_KEYS_TABLE}") as cursor:
            return [
                {
                    KEY: row[0],
                    BALANCE: row[1],
                    RATE_LIMIT_PER_MINUTE: row[2],
                }
                if row
                else None
                for row in await cursor.fetchall()
            ]


async def get_all_logs_for_key(conn: aiosqlite.Connection, api_key: str) -> List[Dict[str, Any]]:
    async with db_lock:
        async with conn.execute(
            f"SELECT {KEY}, {ENDPOINT}, {COST}, {BALANCE}, {CREATED_AT} FROM {LOGS_TABLE} WHERE {KEY} = ? ORDER BY {CREATED_AT} DESC LIMIT 100",
            (api_key,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    KEY: row[0],
                    ENDPOINT: row[1],
                    COST: row[2],
                    BALANCE: row[3],
                    CREATED_AT: row[4],
                }
                for row in rows
            ]


async def get_all_logs(conn: aiosqlite.Connection) -> List[Dict[str, Any]]:
    async with db_lock:
        async with conn.execute(
            f"SELECT {KEY}, {ENDPOINT}, {COST}, {BALANCE}, {CREATED_AT} FROM {LOGS_TABLE}"
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    KEY: row[0],
                    ENDPOINT: row[1],
                    COST: row[2],
                    BALANCE: row[3],
                    CREATED_AT: row[4],
                }
                for row in rows
            ]


async def add_api_key(
    conn: aiosqlite.Connection,
    api_key: str,
    balance: float,
    rate_limit_per_minute: int,
    name: str,
) -> None:
    async with db_lock:
        await conn.execute(
            f"INSERT INTO {API_KEYS_TABLE} VALUES (?, ?, ?, ?, ?)",
            (api_key, name, balance, rate_limit_per_minute, datetime.now()),
        )
        await conn.commit()


async def update_api_key_balance(conn: aiosqlite.Connection, key: str, balance: float):
    async with db_lock:
        await conn.execute(f"UPDATE {API_KEYS_TABLE} SET {BALANCE} = ? WHERE {KEY} = ?", (balance, key))
        await conn.commit()


async def update_api_key_rate_limit(conn: aiosqlite.Connection, key: str, rate: int):
    async with db_lock:
        await conn.execute(
            f"UPDATE {API_KEYS_TABLE} SET {RATE_LIMIT_PER_MINUTE} = ? WHERE {KEY} = ?",
            (rate, key),
        )
        await conn.commit()


async def update_api_key_name(conn: aiosqlite.Connection, key: str, name: str):
    async with db_lock:
        await conn.execute(f"UPDATE {API_KEYS_TABLE} SET {NAME} = ? WHERE {KEY} = ?", (name, key))
        await conn.commit()


async def delete_api_key(conn: aiosqlite.Connection, api_key: str) -> None:
    async with db_lock:
        await conn.execute(f"DELETE FROM {API_KEYS_TABLE} WHERE {KEY} = ?", (api_key,))
        await conn.commit()


async def update_requests_and_credits(conn: aiosqlite.Connection, api_key_info: Dict[str, Any], cost: float) -> None:
    async with db_lock:
        await conn.execute(
            f"UPDATE api_keys SET {BALANCE} = {BALANCE} - ? WHERE {KEY} = ?",
            (cost, api_key_info[KEY]),
        )
        await conn.commit()


async def log_request(conn: aiosqlite.Connection, api_key_info: Dict[str, Any], path: str, cost: float) -> None:
    async with db_lock:
        balance = api_key_info[BALANCE]
        await conn.execute(
            f"INSERT INTO {LOGS_TABLE} ({KEY}, {ENDPOINT}, {BALANCE}, {CREATED_AT}, cost) VALUES (?, ?, ?, ?, ?)",
            (api_key_info[KEY], path, balance, datetime.now(), cost),
        )
        await conn.commit()


async def rate_limit_exceeded(conn: aiosqlite.Connection, api_key_info: Dict[str, Any]) -> bool:
    async with db_lock:
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        query = f"""
            SELECT *
            FROM {LOGS_TABLE}
            WHERE {KEY} = ? AND {CREATED_AT} >= ?
        """
        async with conn.execute(query, (api_key_info[KEY], one_minute_ago.strftime("%Y-%m-%d %H:%M:%S"))) as cursor:
            recent_logs = await cursor.fetchall()
            return len(recent_logs) >= api_key_info[RATE_LIMIT_PER_MINUTE]
