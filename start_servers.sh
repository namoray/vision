#!/bin/bash

for env_file in .*.env; do
    hotkey=$(basename $env_file .env | cut -c 2-)
    if grep -q 'IS_VALIDATOR=True' $env_file; then
        pm2 delete validating_server_$hotkey
        pm2 start validation/proxy/api_server/asgi.py --name validating_server_$hotkey --interpreter python3 -- --env_file $env_file
    else
        hotkey=$(basename $env_file .env | cut -c 2-)
        pm2 delete mining_server_$hotkey
        pm2 start --name mining_server_$hotkey mining/proxy/run_miner.py --interpreter python3 -- --netuid 51 --logging.debug --env_file $env_file
    fi
done
