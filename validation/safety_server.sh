pm2 delete "safety_server"
pm2 start --name "safety_server" "python validation/validator_safety_server/safety_server.py"
