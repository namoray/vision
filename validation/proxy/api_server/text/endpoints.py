from fastapi import HTTPException
from fastapi.responses import JSONResponse
from models import base_models, synapses, utility_models, request_models
from validation.proxy import get_synapse
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
    synapse = get_synapse.get_synapse_from_body(
        body=body,
        synapse_model=synapses.Chat,
    )

    if synapse.model == utility_models.ChatModels.mixtral.value:
        task = tasks.Task.chat_mixtral
    elif synapse.model == utility_models.ChatModels.llama_3.value:
        task = tasks.Task.chat_llama_3
    else:
        raise HTTPException(status_code=400, detail="Invalid model provided")

    text_generator = await core_validator.make_organic_query(
        synapse=synapse, outgoing_model=base_models.ChatOutgoing, stream=True, task=task
    )
    if isinstance(text_generator, JSONResponse):
        return text_generator

    return StreamingResponse(text_generator, media_type="text/plain")
