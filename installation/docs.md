
# Mining on the beast, full instructions

Note: if you're using a provider such as runpod or vast, make sure you expose the necessary ports first, e.g.:
https://docs.runpod.io/docs/expose-ports#:~:text=If%20your%20pod%20supports%20a,address%20to%20access%20your%20service.
Follow the "Symmetrical port mapping" step :)


## Prerequisites for both mining and validating
I would advise using python 3.9 or 3.10. It is known that python 3.12 currently causes issues.

### Install pm2
```bash
apt update && apt upgrade -y
apt install nodejs npm -y
npm i -g pm2
```
## Installing the beast for miners
Clone the repository, install requirements, get the model, configure api keys, badabing, badaboom: 
```bash
git clone https://github.com/namoray/vision.git
cd vision
pip install -r requirements.txt
pip install -e .
./get_model.sh
```

> :warning: **Prerequisite**: Make sure you have a key registered before you run the program.
```bash
btcli s register --netuid 19 --wallet.name YOUR_WALLET_NAME_HERE --wallet.hotkey YOUR_HOTKEY_NAME_HERE
```

```bash
pm2 start miners/miner.py --interpreter python3 --name 19_miner -- --netuid 19 --wallet.name YOUR_WALLET_NAME_HERE --wallet.hotkey YOUR_HOTKEY_NAME_HERE --logging.debug --axon.port YOUR_PORT_HERE --subtensor.network PICK_ONE_OF_finney/local --neuron.device cuda
```
Feel free to change this as you desire, e.g. changing the subtensor config to use a local subtensor.

> :warning: **Note**: Please make sure you have exposed the correct ports on your gpu / instance (default is 8091)

## Installing the beast for validators
Clone the repository, install requirements, get the model, configure api keys, badabing, badaboom: 
```bash
git clone https://github.com/namoray/vision.git
cd vision
pip install -r requirements.txt
pip install -e .
./get_model.sh
wandb login
```

```bash
pm2 start validators/validator.py --interpreter python3 --name 19_validator -- --netuid 19 --subtensor.network finney --wallet.name YOUR_WALLET_NAME_HERE --wallet.hotkey YOUR_HOTKEY_NAME_HERE --logging.info --neuron.device cuda
```
Feel free to change this as you desire, e.g. changing the subtensor config to use a local subtensor