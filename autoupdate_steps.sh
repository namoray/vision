#!/bin/bash

# THIS FILE CONTAINS THE STEPS NEEDED TO AUTOMATICALLY UPDATE THE REPO ON A TAG CHANGE
# THIS FILE ITSELF MAY CHANGE FROM UPDATE TO UPDATE, SO WE CAN DYNAMICALLY FIX ANY ISSUES

# SOME STEPS ARE COMMENTED OUT IN THIS UPDATE AS THEY ARE NOT NEEDED

pm2 delete "api_server"
pm2 delete "safety_server"
pm2 delete "checking_server"

# ./get_models.sh


# if command -v pip &> /dev/null
# then
#     ### Install the local python environment using pip
#     pip install --upgrade pip
#     pip install -e .
#     pip install -r git_requirements.txt
# else
#     ### Install the local python environment using pip3
#     pip3 install --upgrade pip
#     pip3 install -e .
#     pip3 install -r git_requirements.txt
# fi


{
  # Try making the database
  curl -fsSL -o /usr/local/bin/dbmate https://github.com/amacneil/dbmate/releases/latest/download/dbmate-linux-amd64
  chmod +x /usr/local/bin/dbmate
  dbmate --url "sqlite:validator_database.db" up
} || {
  # Except some problemo
  echo "Error occurred while executing the commands."
}


./validation/run_all_servers.sh
