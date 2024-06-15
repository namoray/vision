"""
This is a utility script for manually getting hotkey scores.

Usage:
python get_hotkey_scores.py --env_file {your_vali_hotkey_env_file_here}
"""
from validation.core_validator import core_validator
from validation.weight_setting import calculations
import asyncio
import json
import matplotlib.pyplot as plt

async def main():
    await core_validator.resync_metagraph()
    total_scores = calculations.calculate_scores_for_settings_weights(
        capacities_for_tasks=core_validator.capacities_for_tasks,
        uid_to_uid_info=core_validator.uid_to_uid_info,
        task_weights=core_validator.task_weights
    )
    return total_scores

if __name__ == "__main__":
    result = asyncio.run(main())
    with open("hotkey_scores.json", "w") as file:
        json.dump(result, file)
    scores = list(result.values())
    plt.scatter(range(len(scores)), scores)
    plt.show()