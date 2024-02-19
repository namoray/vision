# Full instructions for setup


### Hardware
Setups tested for mining so far:

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

## Setup steps

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

### Define the configuration for your miners to run
Since you have already installed vision (with `pip install -e .`), you can now use the Vision cli to define the configuration you want.


```bash
vision create-config
```

This will automagically spit out a config.yaml file with all your preferences. You can use
```bash
nano config.yaml
```
at any time to edit it, or `cat config.yaml` to view it! (If you didn't install nano, you can try `nano config.yaml`)

It will also create a start_miners.sh script to automatically run the miners according the config you generated.

**If you change the config.yaml, you will also need to edit the start_miners.sh script to reflect this!**

### Start miners
You can either use the command
```bash
./start_miners.sh
```
Or you can use the usual pm2 commands
```bash
pm2 start --name NAME_FOR_MINER_HERE mining/run_miner.py --interpreter python3 -- --axon.port YOUR_AXON_PORT --axon.external_ip EXTERNAL_IP_FOR_AXON --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOTKEY --subtensor.network SUBTENSOR_NETWORK --netuid 19 --logging.debug
```

**NOTE**
If you want to run multiple miners on the same machine, I would advise just running one first, to let it download all the necessary models and configuration, then start your others up :)
