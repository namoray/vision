#!/bin/bash

set -x

for env_file in .*.env; do
    if grep -q 'IS_VALIDATOR=True' $env_file; then
        hotkey=$(basename $env_file .env | cut -c 2-)
        pm2 delete validating_server_$hotkey
        pm2 start validation/proxy/api_server/asgi.py --name validating_server_$hotkey --interpreter python -- --env_file $env_file 2> errorlog.txt
    fi
done