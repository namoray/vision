import os
import subprocess
import time
from typer import Typer

app = Typer()


def should_update_local(local_tag, remote_tag):
    if remote_tag[0] == local_tag[0]:
        return remote_tag != local_tag
    return False


def run_servers(start_servers_script="run_all_servers.sh"):
    subprocess.call(["./validation/" + start_servers_script], shell=True)


def update_local(remote_tag, start_servers_script):
    reset_cmd = 'git reset --hard ' + remote_tag
    process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print("Error in updating:", error)
    else:
        print("Updated local repo to latest version: {}", format(remote_tag))

        print("Running the autoupdate steps...")
        # Trigger shell script. Make sure this file path starts from root
        if start_servers_script == "run_all_servers.sh":
            autoupdate_script = "./autoupdate_steps.sh"  # So we don't break legacy running servers
        else:
            autoupdate_script = "./autoupdate_" + start_servers_script
        subprocess.call([autoupdate_script], shell=True)

        print("Finished running the autoupdate steps! Ready to go 😎")


@app.command()
def run_and_keep_updated(start_servers_script: str = "run_all_servers.sh"):
    run_servers(start_servers_script)
    while True:
        local_tag = subprocess.getoutput('git describe --abbrev=0 --tags')
        os.system('git fetch')
        remote_tag = subprocess.getoutput('git describe --tags `git rev-list --tags --max-count=1`')

        if should_update_local(local_tag, remote_tag):
            print("Local repo is not up-to-date. Updating...")
            update_local(remote_tag=remote_tag, start_servers_script=start_servers_script)
        else:
            print("Repo is up-to-date.")

        time.sleep(60)


if __name__ == "__main__":
    app()
