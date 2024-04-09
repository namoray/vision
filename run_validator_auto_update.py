import os
import subprocess
import time


def should_update_local(local_tag, remote_tag):
    if remote_tag[0] == local_tag[0]:
        return remote_tag != local_tag
    return False


process = subprocess.Popen(["./launch_validators.sh"], stdout=subprocess.PIPE)
branch_name = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")


def run_auto_updater():
    while True:
        local_tag = subprocess.getoutput("git describe --abbrev=0 --tags")
        os.system(f"git fetch origin {branch_name}")
        remote_tag = subprocess.getoutput(
            f"git describe --tags `git rev-list --tags=origin/{branch_name} --max-count=1`"
        )

        if should_update_local(local_tag, remote_tag):
            print("Local repo is not up-to-date. Updating...")
            reset_cmd = "git reset --hard " + remote_tag
            process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error:
                print("Error in updating:", error)
            else:
                print("Updated local repo to latest version: {}", format(remote_tag))

                print("Running the autoupdate steps...")
                # Trigger shell script. Make sure this file path starts from root
                subprocess.call(["./autoupdate_validator_steps.sh"], shell=True)

                print("Finished running the autoupdate steps! Ready to go 😎")

        else:
            print("Repo is up-to-date.")

        time.sleep(60)


if __name__ == "__main__":
    run_auto_updater()
