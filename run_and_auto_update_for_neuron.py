import argparse
import subprocess
import time
from datetime import datetime
from typing import List

from core import utils

pm2_start_command = None

# Please ADD you webhook url here if you want to be sent discord alerts instead of auto updating
webhook_url = ""


def start_miner(neuron_pm2_name: str, validator: bool, neuron_args: List[str]):
    global pm2_start_command
    if not validator:
        print("Starting miner!")
        pm2_start_command = [
            "pm2",
            "start",
            "miners/miner.py",
            "--interpreter",
            "python3",
            "--name",
            neuron_pm2_name,
        ] + neuron_args
    else:
        print("Starting validator!")
        pm2_start_command = [
            "pm2",
            "start",
            "validators/validator.py",
            "--interpreter",
            "python3",
            "--name",
            neuron_pm2_name,
        ] + neuron_args

    subprocess.run(pm2_start_command, check=True)
    print(f"Started neuron with PM2 under the name: {neuron_pm2_name}")


def check_for_updates_and_restart(neuron_pm2_name: str, check_interval: float, only_alert: bool):
    still_needs_restart = False
    while True:
        try:
            print("Checking for updates... ⏳")
            subprocess.run(["git", "fetch"], check=True)

            local = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
            remote = subprocess.check_output(["git", "rev-parse", "@{u}"]).strip()
            if local != remote or still_needs_restart:
                if only_alert:
                    now = datetime.now()
                    nice_date = now.strftime("%m/%d/%Y, %H:%M:%S")
                    if webhook_url != "":
                        utils.send_discord_alert(
                            message=f"Please update your neuron on subnet 19, you're out of date! Date: {nice_date}",
                            webhook_url=webhook_url,
                        )
                    else:
                        print(f"Please update your neuron on subnet 19, you're out of date! Date: {nice_date}")
                    time.sleep(60 * 30)
                    continue

                print("Changes detected. Pulling updates.")
                try:
                    subprocess.run(["git", "reset", "--hard"])
                    subprocess.run(["git", "pull"])
                    subprocess.run(["pip", "install", "-r", "requirements.txt"])
                    subprocess.run(["pip", "install", "-e", "."])
                    subprocess.run(["pm2", "stop", neuron_pm2_name], check=True)
                    time.sleep(2)
                    subprocess.run(["pm2", "start", neuron_pm2_name], check=True)
                    still_needs_restart = False
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while restarting the PM2 process: {e}. Gonna try again")
                    still_needs_restart = True
            else:
                print("No changes detected, up to date ✅")

            time.sleep(check_interval)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while checking for updates: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor a Git repository for updates and restart a PM2 process if updates are found.",
        epilog='Example usage: pm2 start --name "default-miner-auto-updater" run_and_auto_update.py --interpreter python3 -- --neuron_pm2_name "default-miner"  -- --netuid 19 --wallet.name default  --wallet.hotkey default --logging.debug --axon.port 8091 --subtensor.network finney --neuron.device cuda',
    )
    parser.add_argument(
        "--neuron_pm2_name", required=True, help="Name of the PM2 process for the miner/validator neuron"
    )
    parser.add_argument(
        "--check_interval", type=int, default=60, help="Interval in seconds to check for updates (default: 60)."
    )
    parser.add_argument(
        "--validator",
        action="store_true",
        help="Whether we are running a validator or miner. True if running a validator",
    )
    parser.add_argument(
        "--only_alert", action="store_true", help="Whether to just alert instead of running the validator"
    )
    parser.add_argument("neuron_args", nargs=argparse.REMAINDER, help="Arguments to pass to the miner script")
    args = parser.parse_args()

    start_miner(args.neuron_pm2_name, args.validator, args.neuron_args)
    check_for_updates_and_restart(args.neuron_pm2_name, args.check_interval, args.only_alert)


if __name__ == "__main__":
    main()
