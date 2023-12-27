import argparse
import subprocess
import time
import os

pm2_start_command = None
lock_file_path = 'lock-file.lock'

def acquire_lock(timeout=300):
    start_time = time.time()
    try:
        with os.fdopen(os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY), 'w') as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False

            
def release_lock():
    try:
        os.remove(lock_file_path)
    except FileNotFoundError:
        pass

    
def start_miner(neuron_pm2_name, validator, neuron_args):

    global pm2_start_command
    if not validator:
        print("Starting miner!")
        pm2_start_command = ["pm2", "start", "miners/miner.py", "--interpreter", "python3", "--name", neuron_pm2_name] + neuron_args
    else:
        print("Starting validator!")
        pm2_start_command = ["pm2", "start", "validators/validator.py", "--interpreter", "python3", "--name", neuron_pm2_name] + neuron_args

    subprocess.run(pm2_start_command, check=True)
    print(f"Started neuron with PM2 under the name: {neuron_pm2_name}") 


def check_for_updates_and_restart(neuron_pm2_name, check_interval):
    while True:
        try:
            print("Checking for updates... ⏳")
            subprocess.run(["git", "fetch"], check=True)


            local = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
            remote = subprocess.check_output(["git", "rev-parse", "@{u}"]).strip()


            if local != remote and acquire_lock():
                
                print("Changes detected. Pulling updates.")
                subprocess.run(["git", "reset", "--hard"])
                subprocess.run(["git", "pull"])
                subprocess.run(["pip", "install", "-r", "requirements.txt"])
                subprocess.run(["pip", "install", "-e", "."])
                subprocess.run(["pm2", "delete", neuron_pm2_name], check=True)
                time.sleep(2)
                subprocess.run(pm2_start_command, check=True)
            else:
                print("No changes detected, up to date ✅")
            release_lock()
            time.sleep(check_interval)
 
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while checking for updates: {e}")
            release_lock()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            release_lock()
 
def main():
    parser = argparse.ArgumentParser(
        description="Monitor a Git repository for updates and restart a PM2 process if updates are found.",             
        epilog='Example usage: pm2 start --name "default-miner-auto-updater" run_and_auto_update.py --interpreter python3 -- --neuron_pm2_name "default-miner"  -- --netuid 19 --wallet.name default  --wallet.hotkey default --logging.debug --axon.port 8091 --subtensor.network finney --neuron.device cuda')
    parser.add_argument("--neuron_pm2_name", required=True, help="Name of the PM2 process for the miner/validator neuron")
    parser.add_argument("--check_interval", type=int, default=240, help="Interval in seconds to check for updates (default: 240).")
    parser.add_argument("--validator", action='store_true', help="Whether we are running a validator or miner. True if running a validator")
    parser.add_argument('neuron_args', nargs=argparse.REMAINDER, help="Arguments to pass to the miner script")
    args = parser.parse_args()

    start_miner(args.neuron_pm2_name, args.validator, args.neuron_args)

    # Monitor the repository for updates
    check_for_updates_and_restart(args.neuron_pm2_name, args.check_interval)

if __name__ == "__main__":
    main()
