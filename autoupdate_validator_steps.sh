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

sqlite3 "vision_database.db" "DELETE FROM uid_records;"
./launch_validators.sh
echo "Nothing to do!"