#!/bin/bash

for env_file in .*.env; do
    if ! grep -q 'IS_VALIDATOR=True' $env_file; then
        hotkey=$(basename $env_file .env | cut -c 2-)
        pm2 delete mining_server_$hotkey
        pm2 start --name mining_server_$hotkey mining/proxy/run_miner.py --interpreter python3 -- --logging.debug --env_file $env_file
    fi
done
