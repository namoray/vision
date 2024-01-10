pm2 delete "checking_server"
pm2 start --name "checking_server" "python validation/validator_checking_server/checking_server.py"
