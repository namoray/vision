import os
import subprocess
import time


def should_update_local(local_tag, remote_tag):
    if remote_tag[0] == local_tag[0]:
        return remote_tag != local_tag
    return False

subprocess.call(["./validation/run_all_servers.sh"], shell=True)

while True:

    local_tag = subprocess.getoutput('git describe --abbrev=0 --tags')
    os.system('git fetch')
    remote_tag = subprocess.getoutput('git describe --tags `git rev-list --tags --max-count=1`')

    if should_update_local(local_tag, remote_tag):
        print("Local repo is not up-to-date. Updating...")
        update_cmd = 'git pull origin ' + remote_tag
        process = subprocess.Popen(update_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print("Error in updating:", error)
        else:
            print("Updated local repo to latest version: {}", format(remote_tag))

            print("Starting the validator servers...")
            # Trigger shell script. Make sure this file path starts from root
            subprocess.call(["./validation/run_all_servers.sh"], shell=True)

            print("Finished starting all the validator servers! Ready to go 😎")

    else:
        print("Repo is up-to-date.")

    time.sleep(60)
