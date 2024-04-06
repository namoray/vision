from typing import Dict, List, Optional

import bittensor as bt

from models import base_models

from typing import AsyncIterator

from starlette.responses import StreamingResponse


class AvailableTasksOperation(bt.Synapse, base_models.AvailableTasksOperationBase):
    def deserialize(self) -> Optional[Dict[str, bool]]:
        return self.available_tasks


class TextToImage(bt.Synapse, base_models.TextToImageBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64


class ImageToImage(bt.Synapse, base_models.ImageToImageBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64


class Inpaint(bt.Synapse, base_models.InpaintBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64

class Avatar(bt.Synapse, base_models.AvatarBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64
    
# class Scribble(bt.Synapse, base_models.ScribbleBase):
#     def deserialize(self) -> Optional[List[str]]:
#         return self.image_b64


class Upscale(bt.Synapse, base_models.UpscaleBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64


class ClipEmbeddings(bt.Synapse, base_models.ClipEmbeddingsBase):
    def deserialize(self) -> Optional[List[List[float]]]:
        return self.clip_embeddings


class Sota(bt.Synapse, base_models.SotaBase):
    def deserialize(self) -> Optional[str]:
        return self.image_url


class Chat(bt.StreamingSynapse, base_models.ChatBase):
    def deserialize(self) -> Optional[Dict[str, str]]:
        return None

    async def process_streaming_response(
        self, response: StreamingResponse
    ) -> AsyncIterator[str]:
        async for chunk in response.content.iter_any():
            if isinstance(chunk, bytes):
                tokens = chunk.decode("utf-8")
                # TODO: Wrap up into a payload?
                yield tokens

    def extract_response_json(self, response) -> dict:
        """
        Abstract method that must be implemented by the subclass.
        This method should provide logic to extract JSON data from the response, including headers and content.
        It is called after the response has been processed and is responsible for retrieving structured data
        that can be used by the application.

        Args:
            response: The response object from which to extract JSON data.
        """
        return {}
