CREATE TABLE tasks (
                task_name TEXT,
                checking_data TEXT
            );
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
CREATE TABLE scores
(
    id INTEGER PRIMARY KEY,
    axon_uid INT NOT NULL,
    hotkey TEXT NOT NULL,
    response_time FLOAT,
    score FLOAT,
    synapse TEXT,
    valid_response BOOLEAN,
    quickest_response BOOLEAN,
    checked_with_server BOOLEAN,
    timestamp TIMESTAMP
);
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20240213111625'),
  ('20240304133322');
