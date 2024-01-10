pm2 delete "api_server"
if [ $# -gt 0 ]; then
    pm2 start validation/validator_api_server/api_server.py --name api_server --interpreter python -- "$@"
else
    pm2 start validation/validator_api_server/api_server.py --name api_server --interpreter python
fi
