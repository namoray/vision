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

# dbmate --url "sqlite:vision_database.db" up

# pip install -e .
# ./launch_validators.sh
if [ -f "validator_database.db" ]; then
    rm "validator_database.db"
    echo "Deleted validator database"
else
    echo "Nothing to do!"
fi
