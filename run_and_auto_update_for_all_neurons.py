import argparse
import subprocess
import time


def check_for_updates_and_restart(check_interval):
    still_needs_restart = False
    while True:
        try:
            print("Checking for updates... ⏳")
            subprocess.run(["git", "fetch"], check=True)

            local = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
            remote = subprocess.check_output(["git", "rev-parse", "@{u}"]).strip()

            if local != remote or still_needs_restart:
                print("Changes detected. Pulling updates.")
                try:
                    subprocess.run(["git", "reset", "--hard"])
                    subprocess.run(["git", "pull"])
                    subprocess.run(["pip", "install", "-r", "requirements.txt"])
                    subprocess.run(["pip", "install", "-e", "."])
                    subprocess.run(["pm2", "restart", "all"], check=True)
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
        description="Monitor a Git repository for updates and restart all PM2 processes if updates are found.",
        epilog='Example usage: pm2 start --name "run_auto_updates_for_all_neurons" run_and_auto_update_for_all_neurons.py --interpreter python3',
    )

    parser.add_argument(
        "--check_interval", type=int, default=60, help="Interval in seconds to check for updates (default: 60)."
    )
    args = parser.parse_args()
    check_for_updates_and_restart(args.check_interval)


if __name__ == "__main__":
    main()
