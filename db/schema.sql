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
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20240213111625'),
  ('20240304133322'),
  ('20240602214214'),
  ('20240602225958'),
  ('20240602230138'),
  ('20240603230153');
