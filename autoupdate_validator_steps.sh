#!/bin/bash

# THIS FILE CONTAINS THE STEPS NEEDED TO AUTOMATICALLY UPDATE THE REPO ON A TAG CHANGE
# THIS FILE ITSELF MAY CHANGE FROM UPDATE TO UPDATE, SO WE CAN DYNAMICALLY FIX ANY ISSUES


# OLD_DB="validator_database.db"
# NEW_DB="vision_database.db"

# # Check if the old database file exists
# if [ -f "$OLD_DB" ]; then
#     # Rename the old database file to the new name
#     mv "$OLD_DB" "$NEW_DB"
#     echo "Renamed $OLD_DB to $NEW_DB"
# else
#     echo "$OLD_DB does not exist"
# fi


# pip install -e .
#!/bin/bash

# Function to run a command with sudo only if it is available
run_if_sudo_available() {
    local cmd="$1"
    if command -v sudo &>/dev/null; then
        sudo bash -c "$cmd"
    else
        bash -c "$cmd"
    fi
}


dbmate --url "sqlite:vision_database.db" up
run_if_sudo_available "apt install sqlite3"
sqlite3 "vision_database.db" "DELETE FROM uid_records;"
./launch_validators.sh
echo "Nothing to do!"