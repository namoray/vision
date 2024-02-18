# Full instructions for setup


## Prerequisites for both mining and validating
I would advise using python 3.9 or 3.10. It is known that python 3.12 currently causes issues.


```bash
### clone the repo
git clone https://github.com/namoray/vision
cd vision
```


```bash
### Install pm2 & jq
apt update && apt upgrade -y
apt install nodejs npm -y
npm i -g pm2
apt-get install -y jq

### Install vim so you can edit the config easily
apt-get update
apt-get install vim

```bash
### Install the local python environment
pip install --upgrade pip
pip install -e .
pip install -r git_requirements.txt
```

### Download the models
```bash
./get_models.sh
```

```bash
### Create config
vision create-config
```


Creating the config allows you to specify which devices you want each GPU model to run on, and automatically makes the start_miners.sh script for you.

How you distribute the models is up to you.

#### Extra guidance for Validators
If you have a machine with >=24gb vram (e.g. a 4090, a100, h100 etc), I would advise just using '0' for each of the device settings.

The default CHECKING_SERVER_ADDRESS and SAFETY_CHECKER_SERVER_ADDRESS will work fine - it will all run on one machine


## Miners:
Check you're happy with the config and edit the ./start_miners.sh file to your liking


Note if you're running multiple miners on one system, for the very first start on this system, I would advise just starting one miner first. This is because lots of downloads of models and stuff are needed. If you run
```bash
cat start_miners.sh
```

And just use one of those commands first, then you should be golden :)

After, you can run

```bash
./start_miners.sh
```

## Validators

### With auto updates
NOTE: Auto updates only with automatically update non major versions - it won't update 2.x.x to 3.x.x.
