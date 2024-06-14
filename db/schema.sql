CREATE TABLE IF NOT EXISTS "schema_migrations" (version varchar(128) primary key);
CREATE TABLE api_keys (
    key TEXT PRIMARY KEY,
    name TEXT,
    balance REAL,
    rate_limit_per_minute INTEGER DEFAULT 60,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE logs (
    key TEXT,
    endpoint TEXT,
    cost REAL,
    balance REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(key) REFERENCES api_keys(key) ON DELETE CASCADE
);
CREATE TABLE reward_data (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    task TEXT NOT NULL,
    axon_uid INTEGER NOT NULL,
    quality_score FLOAT NOT NULL,
    validator_hotkey TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    synthetic_query BOOLEAN NOT NULL,
    speed_scoring_factor FLOAT,
    response_time FLOAT,
    volume FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    checking_data TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE uid_records (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    axon_uid INT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    validator_hotkey TEXT NOT NULL,
    task TEXT NOT NULL,
    declared_volume FLOAT NOT NULL,
    consumed_volume FLOAT DEFAULT 0,
    total_requests_made INT DEFAULT 0,
    requests_429 INT DEFAULT 0,
    requests_500 INT DEFAULT 0,
    period_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE miner_concurrency_group (
    concurrency_group_id INTEGER PRIMARY KEY AUTOINCREMENT,
    concurrent_tasks_limit INTEGER NOT NULL,
    name TEXT
);
CREATE TABLE miner_task_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    volume FLOAT NOT NULL,
    concurrency_group_id INTEGER,
    FOREIGN KEY (concurrency_group_id) REFERENCES miner_concurrency_group(concurrency_group_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX idx_api_keys_key_unique ON api_keys(key);
CREATE INDEX idx_logs_key ON logs(key);
CREATE INDEX idx_logs_endpoint ON logs(endpoint);
CREATE INDEX idx_logs_created_at ON logs(created_at);
CREATE INDEX idx_reward_data_task ON reward_data(task);
CREATE INDEX idx_reward_data_axon_uid ON reward_data(axon_uid);
CREATE INDEX idx_reward_data_quality_score ON reward_data(quality_score);
CREATE INDEX idx_reward_data_validator_hotkey ON reward_data(validator_hotkey);
CREATE INDEX idx_reward_data_miner_hotkey ON reward_data(miner_hotkey);
CREATE INDEX idx_reward_data_created_at ON reward_data(created_at);
CREATE INDEX idx_tasks_task_name ON tasks(task_name);
CREATE INDEX idx_tasks_miner_hotkey ON tasks(miner_hotkey);
CREATE INDEX idx_tasks_created_at ON tasks(created_at);
CREATE INDEX idx_uid_records_task ON uid_records(task);
CREATE INDEX idx_uid_records_axon_uid ON uid_records(axon_uid);
CREATE INDEX idx_uid_records_validator_hotkey ON uid_records(validator_hotkey);
CREATE INDEX idx_uid_records_miner_hotkey ON uid_records(miner_hotkey);
CREATE INDEX idx_uid_records_created_at ON uid_records(created_at);
CREATE INDEX idx_miner_task_config_task_name ON miner_task_config(task_name);
CREATE INDEX idx_miner_task_config_miner_hotkey ON miner_task_config(miner_hotkey);
CREATE INDEX idx_miner_task_config_concurrency_group_id ON miner_task_config(concurrency_group_id);
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20240213111625'),
  ('20240304133322'),
  ('20240602214214'),
  ('20240602225958'),
  ('20240602230138'),
  ('20240603230153'),
  ('20240611144515'),
  ('20240614111428');
