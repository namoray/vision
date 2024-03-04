-- migrate:up

CREATE TABLE scores
(
    id SERIAL PRIMARY KEY,
    axon_uid INT NOT NULL,
    hotkey TEXT,
    response_time FLOAT,
    score FLOAT,
    synapse TEXT,
    valid_response BOOLEAN,
    quickest_response BOOLEAN,
    checked_with_server BOOLEAN,
    timestamp TIMESTAMP
);



-- migrate:down

DROP TABLE scores;

