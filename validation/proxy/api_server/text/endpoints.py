from fastapi import HTTPException
from models import base_models, synapses, utility_models, request_models
from validation.proxy import validation_utils
from starlette.responses import StreamingResponse
from core import tasks
from fastapi.routing import APIRouter
from validation.core_validator import core_validator
import fastapi
from validation.proxy import dependencies

router = APIRouter()


@router.post("/chat")
async def chat(
    body: request_models.ChatRequest,
    _: None = fastapi.Depends(dependencies.get_token),
) -> StreamingResponse:
    synapse = validation_utils.get_synapse_from_body(
        body=body,
        synapse_model=synapses.Chat,
    )

    if synapse.model == utility_models.ChatModels.bittensor_finetune.value:
        task = tasks.Tasks.chat_bittensor_finetune.value
    elif synapse.model == utility_models.ChatModels.mixtral.value:
        task = tasks.Tasks.chat_mixtral.value
    else:
        raise HTTPException(status_code=400, detail="Invalid model provided")

    text_generator = await core_validator.execute_query(
        synapse, outgoing_model=base_models.ChatOutgoing, stream=True, task=task, synthetic_query=False
    )
    return StreamingResponse(text_generator, media_type="text/plain")
