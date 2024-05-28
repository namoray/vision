from models import base_models
import bittensor as bt
import httpx
import json
from typing import AsyncGenerator

POST_ENDPOINT = "generate_text"


async def stream_text_from_server(body: base_models.ChatIncoming, url: str):
    text_endpoint = url + POST_ENDPOINT
    async with httpx.AsyncClient() as client:  # noqa
        async with client.stream("POST", text_endpoint, json=body.dict()) as resp:
            async for chunk in resp.aiter_lines():
                try:
                    received_event_chunks = chunk.split("\n\n")
                    for event in received_event_chunks:
                        if event == "":
                            continue
                        prefix, _, data = event.partition(":")
                        if data.strip() == "[DONE]":
                            break
                        loaded_chunk = json.loads(data)
                        text = loaded_chunk["text"]
                        logprob = loaded_chunk["logprobs"][0]["logprob"]
                        data = json.dumps({"text": text, "logprob": logprob})

                        yield f"data: {data}\n\n"
                except Exception as e:
                    bt.logging.error(
                        f"Error in streaming text from the server: {e}. Original chunk: {chunk}"
                    )


async def chat_logic(body: base_models.ChatIncoming, url: str) -> AsyncGenerator:
    """Add gpu potential"""

    text_generator = stream_text_from_server(body, url)
    bt.logging.info(f"✅ About do some chatting, with model {body.model}✨")
    return text_generator
