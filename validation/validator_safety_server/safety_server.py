from fastapi import FastAPI

from core import constants as cst, utils
from core import resource_management
from validation.validator_safety_server.endpoints.safety import router as safety_router
from typing import Dict, Any
import yaml
import uvicorn


app = FastAPI(debug=False)

yaml_config: Dict[str, Any] = yaml.safe_load(open(cst.CONFIG_FILEPATH))
validator_hotkey_name = utils.get_validator_hotkey_name_from_config(yaml_config)


resource_management.set_hotkey_name(validator_hotkey_name)
resource_management.SingletonResourceManager().load_config()
resource_management.SingletonResourceManager().load_resource(cst.IMAGE_SAFETY_CHECKERS)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(safety_router)

server_address = yaml_config.get(validator_hotkey_name, {}).get('SAFETY_CHECKER_SERVER_ADDRESS', 'http://127.0.0.1:9001/').replace('http://', '')
host, port_with_slash = server_address.split(':')
port = port_with_slash[:-1]

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=int(port), loop="asyncio", log_level="debug")
