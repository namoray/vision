"""
This is a utility script for manually setting weights in the case of any issues
Use cases:
- Testing weight setting to see scores and help debug
- Emergencies if weight setting in the vali is nothing working [not recommended]

NOTE: this is not artificial weights, it uses real values obtained by the validator proxy only.
It's just taking that part of the code, and making it runnable

Usage:
python manually_set_weights.py --env_file {your_vali_hotkey_env_file_here}
"""
from validation.core_validator import core_validator
from validation.weight_setting import calculations

core_validator.resync_metagraph()
total_scores = calculations.calculate_scores_for_settings_weights(
    capacities_for_tasks=core_validator.capacities_for_tasks,
    uid_to_uid_info=core_validator.uid_to_uid_info,
    task_weights=core_validator.task_weights
)
print(total_scores)