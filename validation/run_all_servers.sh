#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

"$SCRIPT_DIR/api_server.sh" "$@"

"$SCRIPT_DIR/checking_server.sh"

"$SCRIPT_DIR/safety_server.sh"
