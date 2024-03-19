# Full instructions for setup


### Hardware
Setups tested for mining so far:


**NOTE** Some GPUS (especially on runpod) are unable to load the pipelines into memory.

If you see an issue where the checking servers / safety servers are unable to start, chances are your GPU is incompatible.

There is a bug with the hugging face diffusers library and certain GPU's.

| Name  | CUDA Version | Ubuntu Version | Python Version | Works |
|-------|--------------|----------------|----------------|-------|
| H1OO  | 12.0 | 22.04 | 3.10.12 | ✅  |
| A100 | 11.8  | 22.04 | 3.10.12 | ✅ / ❌ (hit and miss) |
| RTX 4090 | 11.8  | 22.04 | 3.10.12 | ✅ |
| A6000* | 12.0   | 22.04 | 3.10.12 |✅ |
| A40 | 12.0   | 22.04 | 3.10.12 | ✅ |
| L40 | 12.0   | 22.04 | 3.10.12 | ❌ |
| A100 SXM | 11.8  | 22.04 | 3.10.12 | ❌|

Note: That's not to say you can't use other GPU's!

* Recommended - I advise you to use a A6000

### Validators
GPU: 1x80 GB or 1x24GB or 1x48GB

Storage: 200Gb


# Setup steps

Note: if you're using a provider such as runpod or vast, make sure you expose the necessary ports first, e.g.: https://docs.runpod.io/docs/expose-ports#:~:text=If%20your%20pod%20supports%20a,address%20to%20access%20your%20service. Follow the "Symmetrical port mapping" step :)

### Clone the repo
```bash
git clone https://github.com/namoray/vision
cd vision
```


### Install system dependencies

If you are in a container such as runpod, run these:

```bash
### Install pm2 & jq
apt update && apt upgrade -y
apt install nodejs npm -y
npm i -g pm2
apt-get install -y jq

### Install nano so you can edit the config easily
apt-get update
apt-get install nano

### Install lsof so you can see ports not in use
apt-get install lsof
```

If you are on a bare metal machine (e.g. Vast) where you require `sudo`, use the following:
```bash
### Install pm2 & jq
sudo apt update && apt upgrade -y
sudo apt install nodejs npm -y
sudo npm i -g pm2
sudo apt-get install -y jq

### Install nano so you can edit the config easily
sudo apt-get update
sudo apt-get install nano

### Install lsof so you can see ports not in use
sudo apt-get install lsof
```


### Install python dependencies
Make sure you have installed the correct python version, and then follow these steps:

```bash
### Install the local python environment
pip install --upgrade pip
pip install -e .
pip install -r git_requirements.txt
```

If for some reason that doesn't work, you may need to use `pip3`;
```bash
### Install the local python environment
pip3 install --upgrade pip
pip3 install -e .
pip3 install -r git_requirements.txt
```

### Download the necessary models
Simple step this one - note it may take a while if you have poor bandwidth (not recommended)

```bash
./get_models.sh
```

## Bandwidth selling validator
This is the less simple but more exciting of the two! Now you get to sell your bandwidth to whoever you want, with a very simple to use CLI!

The steps are similar to synthetic only validators:

**If you are renting a virtualized GPU e.g. on runpod:**

Before you make the config, run the command
```bash
lsof -i -P -n | grep LISTEN
```

This shows you all the ports currently in use. When you come to choose a `CHECKING_SERVER_ADDRESS` and `SAFETY_CHECKER_SERVER_ADDRESS`, make sure the ports you choose aren't already in use.

Example output:
![image](images/ports_in_use.png)

This shows that ports 9091, 3001, .., 34579, 41133 etc, are currently in use, so pick address that don't include these.

Typically the default values I provide will work :)


#### Creating the config

```bash
vision create-config
```

Make sure to use a port which is exposed to the outside world for API_SERVER_PORT.

Note: if you're using a provider such as runpod, there are instructions here https://docs.runpod.io/docs/expose-ports#:~:text=If%20your%20pod%20supports%20a,address%20to%20access%20your%20service. Follow the "Symmetrical port mapping" step :)

#### Starting the server

### With autoupdates

> **WARNING:** Autoupdates will work best if you use the `one gpu for all` setting in the config - otherwise if there is a new model we won't know what to do!

```bash
pm2 start --name run_validator_auto_update "python run_validator_auto_update.py"
```

### Without auto updates
```bash
./validation/run_all_servers.sh
```

This uses the config defined in the `config.yaml`.

### Running the API server on a separate host - without auto updates

With the methods mentioned above, all components of the validator (API server, checking server and safety server) run on the same host. It is also possible to run the API server on a separate host - even one without a GPU. To run the components separately, follow these steps:

*Step 1: Run checking server and safety server on GPU host, individually*

On the GPU host, edit the config.yaml to make sure the checking server and the safety server run on an exposed port. E.g.: 

    CHECKING_SERVER_ADDRESS: http://0.0.0.0:43308/
    SAFETY_CHECKER_SERVER_ADDRESS: http://0.0.0.0:43309/

Then run the checking server and the safety server manually:

    ./validation/checking_server.sh
    ./validation/safety_server.sh

*Step 2: Run the API server on the non-GPU host*

On the non-GPU host, edit the config.yaml to make sure it points to the GPU host for the checking and safety servers. E.g.: 

    CHECKING_SERVER_ADDRESS: http://141.193.30.26:43308/
    SAFETY_CHECKER_SERVER_ADDRESS: http://141.193.30.26:43309/

Then run the API server manually:

    ./validation/api_server.sh

Use the `pm2 logs` command to make sure the API server has now connected to the remote components on the GPU host. It hits the healthcheck endpoint on each of these services on start-up, to make sure they are online. If there are problems accessing that endpoint, there will be log entries indicating that. If not - the API server should now be accepting requests as normal. 


### Running the API server on a separate host - with auto updates

To run the API server on a separate host with auto updates, follow the same steps as above, but use the `run_validator_auto_update.py` script, passing it the correct starter script as argument. E.g.:

*On the GPU host:*

    pm2 start --name run_validator_workers_auto_update "python run_validator_auto_update.py --start-servers-script run_worker_servers.sh"

*On the non-GPU host:*

    pm2 start --name run_validator_api_auto_update "python run_validator_auto_update.py --start-servers-script api_server.sh"


# Selling your bandwidth

### Creating the database
Used to store api keys & scoring logs
```bash
curl -fsSL -o /usr/local/bin/dbmate https://github.com/amacneil/dbmate/releases/latest/download/dbmate-linux-amd64
chmod +x /usr/local/bin/dbmate

dbmate --url "sqlite:validator_database.db" up
```

### Managing access

To manage access to your api server and sell access to anyone you like, using the vision-cli is the easiest way.


```bash
vision --help
```

Shows all the commands and should give self-explanatory instructions.

You can also do

```bash
vision some-command --help
```

To get more info about that command!

# Examples

For example:

```bash
vision create-key 10 60 test
```
Creates a test key with a balance of 10 (which corresponds to 10 images), a rate limit of 60 requests per minute = 1/s, and a name 'test'.

**Recommend values:**
- Balance: Depends on how much you want to sell! Each credit is an image (so a balance of 1000 will allow 1000 images to be generated)
- Rate limit: I would recommend a rate limit of ~20/minute for casual users trying out the API, and around ~120/minute for production users
- Name: Just for you to remember who you want to use that key :)

Now you can do:
```bash
vision list-keys
```
To see the API key. Give / sell this access to whoever you want to have access to your API server to query the network organically - these will be scored too!

## Allowing people to access your server
For them to use your server, you will need to communicate:

- Your server address (IP_ADDRESS:PORT)
- Use /redoc or /docs for automatic documentation on how to use it!
