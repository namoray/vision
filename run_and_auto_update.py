import argparse
import subprocess
import time

def start_miner(pm2_name, miner_args):
    # Start the miner with PM2
    pm2_start_command = ["pm2", "start", "miners/miner.py", "--interpreter", "python3", "--name", pm2_name] + miner_args
    subprocess.run(pm2_start_command, check=True)
    print(f"Started miner with PM2 under the name: {pm2_name}")

def check_for_updates_and_restart(pm2_name, check_interval):
    while True:
        try:
            # Fetch the latest changes from the remote repository
            print("Checking for updates... ⏳")
            subprocess.run(["git", "fetch"], check=True)

            # Compare the local HEAD to the remote HEAD
            local = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
            remote = subprocess.check_output(["git", "rev-parse", "@{u}"]).strip()

            # If the local commit differs from the remote commit, there are changes
            if local != remote:
                print("Changes detected. Pulling updates.")
                subprocess.run(["git", "pull"], check=True)
                subprocess.run(["pm2", "restart", pm2_name], check=True)
            else:
                print("No changes detected, up to date ✅")

            # Wait for the specified interval before checking again
            time.sleep(check_interval)
 
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while checking for updates: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor a Git repository for updates and restart a PM2 process if updates are found.")
    parser.add_argument("--pm2_name", required=True, help="Name of the PM2 process to restart.")
    parser.add_argument("--check_interval", type=int, default=1200, help="Interval in seconds to check for updates (default: 1200).")
    parser.add_argument('miner_args', nargs=argparse.REMAINDER, help="Arguments to pass to the miner script")
    args = parser.parse_args()

    # Start the miner process
    start_miner(args.pm2_name, args.miner_args)

    # Monitor the repository for updates
    check_for_updates_and_restart(args.pm2_name, args.check_interval)

if __name__ == "__main__":
    main()
