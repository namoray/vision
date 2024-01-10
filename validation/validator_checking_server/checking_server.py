from fastapi import FastAPI

from core import constants as cst, utils
from core import resource_management
from validation.validator_checking_server.endpoints.checking_endpoints import router as checking_router
from validation.validator_checking_server.endpoints.synthetic_generation_endpoints import router as synthetic_router
import yaml
from typing import Dict, Any
import uvicorn
import bittensor as bt

app = FastAPI(debug=False)

@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(checking_router)
app.include_router(synthetic_router)



if __name__ == "__main__":
    yaml_config: Dict[str, Any] = yaml.safe_load(open(cst.CONFIG_FILEPATH))
    validator_hotkey_name = utils.get_validator_hotkey_name_from_config(yaml_config)
    if validator_hotkey_name is None:
        raise ValueError("Please set up the config for a validator!")

    resource_management.set_hotkey_name(validator_hotkey_name)
    singleton = resource_management.SingletonResourceManager()
    singleton.load_config()
    bt.logging.info("Loading all models into RAM")
    for load_function in singleton.resource_name_to_load_function.values():
        load_function()
        singleton.unload_all_models()

    singleton.load_validator_resources()
    singleton.load_resource(cst.MODEL_CACHE)


    server_address = yaml_config.get(validator_hotkey_name, {}).get('CHECKING_SERVER_ADDRESS', 'http://127.0.0.1:8000/').replace('http://', '')
    host, port_with_slash = server_address.split(':')
    port = port_with_slash[:-1]

    uvicorn.run(app, host=host, port=int(port), loop="asyncio", log_level="debug")
