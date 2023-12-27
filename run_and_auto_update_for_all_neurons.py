import argparse
import subprocess
import time
import os

def check_for_updates_and_restart(check_interval):
    while True:
        try:
            print("Checking for updates... ⏳")
            subprocess.run(["git", "fetch"], check=True)

            local = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
            remote = subprocess.check_output(["git", "rev-parse", "@{u}"]).strip()
            if local != remote: 
                print("Changes detected. Pulling updates.")
                subprocess.run(["git", "reset", "--hard"])
                subprocess.run(["git", "pull"])
                subprocess.run(["pip", "install", "-r", "requirements.txt"])
                subprocess.run(["pip", "install", "-e", "."])
                subprocess.run(["pm2", "restart", "all"], check=True)
            else:
                print("No changes detected, up to date ✅")
                
            time.sleep(check_interval)
 
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while checking for updates: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
  
def main():

    check_for_updates_and_restart()

if __name__ == "__main__":
    main()
