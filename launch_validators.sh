#!/bin/bash

set -x

for env_file in .*.env; do
    if grep -q 'IS_VALIDATOR=True' $env_file; then
        hotkey=$(basename $env_file .env | cut -c 2-)
        pm2 delete validating_server_$hotkey

        # Check if netuid is set
        NETUID_OPTION=""
        if [ -n "$netuid" ]; then
            NETUID_OPTION="--netuid $netuid"
        fi

        pm2 start validation/proxy/api_server/asgi.py --name validating_server_$hotkey --interpreter python -- --env_file $env_file $NETUID_OPTION
    fi
done



if [[ ! "$*" == *"--without-cron"* ]]; then
    echo "Setting up cron job with PM2..."
    pm2 delete restart_validator_cron
    pm2 start --name "restart_validator_cron" restart_cron.py --interpreter python3
    echo "Cron job setup complete."
else
    echo "Skipping cron job setup due to --without-cron flag."
fi