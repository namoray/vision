
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
## Installing for miners
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

### To run without auto updates:
```bash
pm2 start miners/miner.py --interpreter python3 --name 19_miner -- --netuid 19 --wallet.name YOUR_WALLET_NAME_HERE --wallet.hotkey YOUR_HOTKEY_NAME_HERE --logging.debug --axon.port YOUR_PORT_HERE --subtensor.network PICK_ONE_OF_finney/local --neuron.device cuda
```
Feel free to change this as you desire, e.g. changing the subtensor config to use a local subtensor.


### To run with auto updates:

#### For one Miner
```bash
pm2 start --name  "default-miner-auto-updater" run_and_auto_update_for_neuron.py --interpreter python3 -- --neuron_pm2_name "default-miner" -- --netuid 19 --wallet.name YOUR_WALLET_NAME_HERE  --wallet.hotkey YOUR_HOTKEY_NAME_HERE  --axon.port YOUR_PORT_GOES_HERE --logging.debug --subtensor.network PICK_ONE_OF_finney/local  --neuron.device cuda
```
Feel free to change this as you desire, e.g. changing the subtensor config to use a local subtensor

#### For more than one miner/validator
Start your miners in the same way, and then run:
```bash
pm2 start --name "run_auto_updates_for_all_neurons" run_and_auto_update_for_all_neurons.py --interpreter python3
```
> :warning: **Note**: Please make sure you have exposed the correct ports on your gpu / instance (default is 8091)

## Installing for validators
Clone the repository, install requirements, get the model, configure api keys, badabing, badaboom: 
```bash
git clone https://github.com/namoray/vision.git
cd vision
pip install -r requirements.txt
pip install -e .
./get_model.sh
wandb login
```

### To run without auto updates:
```bash
pm2 start validators/validator.py --interpreter python3 --name 19_validator -- --netuid 19 --subtensor.network finney --wallet.name YOUR_WALLET_NAME_HERE --wallet.hotkey YOUR_HOTKEY_NAME_HERE --logging.info --neuron.device cuda
```
Feel free to change this as you desire, e.g. changing the subtensor config to use a local subtensor

### To Run with auto updates:
Example command:
```bash
pm2 start --name  "default-validator-auto-updater" run_and_auto_update_for_neuron.py --interpreter python3 -- --neuron_pm2_name "default-validator" --validator -- --netuid 19 --wallet.name YOUR_WALLET_NAME_HERE  --wallet.hotkey YOUR_HOTKEY_NAME_HERE  --logging.debug --subtensor.network finney  --neuron.device cuda
```

Feel free to change this as you desire, e.g. changing the subtensor config to use a local subtensor