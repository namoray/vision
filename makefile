.PHONY:  web-db miners validator config db

config:
	python create_config.py

miners:
	./start_miners.sh

validator:
	./validation/run_all_servers.sh

api_server:
	./validation/api_server.sh

db:
	dbmate --url "sqlite:validator_database.db" up
