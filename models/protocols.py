from typing import Dict, List, Optional, Tuple

import bittensor as bt

from models import base_models


class AvailableOperations(bt.Synapse, base_models.AvailableOperationsBase):
    def deserialize(self) -> Optional[Dict[str, bool]]:
        return self.available_operations


class TextToImage(bt.Synapse, base_models.TextToImageBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class ImageToImage(bt.Synapse, base_models.ImageToImageBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class Inpaint(bt.Synapse, base_models.InpaintBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class Scribble(bt.Synapse, base_models.ScribbleBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class Upscale(bt.Synapse, base_models.UpscaleBase):
    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class ClipEmbeddings(bt.Synapse, base_models.ClipEmbeddingsBase):
    def deserialize(self) -> Optional[List[List[float]]]:

        return self.image_embeddings



class Segment(bt.Synapse, base_models.SegmentBase):
    def deserialize(self) -> Tuple[Optional[List[List[List[int]]]], Optional[List[int]]]:
        """
        Deserialize the embeddings response with masks and image shape
        """
        return self.masks, self.image_shape
