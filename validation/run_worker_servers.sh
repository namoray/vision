#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

"$SCRIPT_DIR/safety_server.sh"
sleep 30
"$SCRIPT_DIR/checking_server.sh"
sleep 30
