from core import resource_management, utils

validator_hotkey_name = utils.get_validator_hotkey_name_from_config()

resource_management.set_hotkey_name(validator_hotkey_name)
singleton = resource_management.SingletonResourceManager()
