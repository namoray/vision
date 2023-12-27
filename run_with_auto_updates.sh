#!/bin/bash

# Function to check for updates
check_for_updates() {
    # Fetch the latest changes from the remote repository
    echo "Checking for updates... ⏳"
    git fetch

    # Compare the local HEAD to the remote HEAD
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse @{u})

    # If the local commit differs from the remote commit, there are changes
    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "Changes detected. Pulling updates."
        git pull
        exit 42
    fi
}

# Interval in seconds between checks for updates (20 minutes)
CHECK_INTERVAL=10

# Start the main Python application and redirect its output to stdout/stderr
python miners/miner.py --netuid 51 --wallet.name default --wallet.hotkey default --logging.debug --axon.port 8091 --subtensor.network test &

# Get the Python application's PID
APP_PID=$!

# Trap SIGTERM to handle script termination
trap "echo 'Stopping script...'; kill $APP_PID; exit 0" SIGTERM

# Loop to check for updates at the interval
while true; do
    # Wait for the specified interval
    sleep "$CHECK_INTERVAL"

    # Check for updates
    check_for_updates
done

# Wait for the Python application to finish (it shouldn't under normal circumstances)
wait $APP_PID
