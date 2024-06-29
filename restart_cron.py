"""
**Restarts the main validation script every X hours**

This will prevent any weird errors that only happen after days of the main process running - especially the process
needs to do so much (resync metagraph, query all miners, store all results, score, etc, etc)

In the very near future, we will have ephemeral 'proxy workers' to do the querying which can be short lived.
We will then have blue/green restarts of the central processor, so there is 0 downtime for any users during maintainence
or updates
"""

import time
import os

TIME_TO_SLEEP = 60 * 60 * 6.5  # 6 and a quarter hours


def main():
    while True:
        print(f"Restarting in {TIME_TO_SLEEP / 60 / 60:.2f} hours")
        time.sleep(TIME_TO_SLEEP)
        os.system("./launch_validators.sh --without-cron")


if __name__ == "__main__":
    main()
