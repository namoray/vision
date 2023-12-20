from typing import List, Optional, Union

import bittensor as bt
from pydantic import Field


class IsAlive(bt.Synapse):
    answer: Optional[str] = None

    def deserialize(self) -> Optional[str]:
        return self.answer


class SegmentingSynapse(bt.Synapse):
    """
    Segment according the given points and boxes for a given image embedding.

    If you dont have the image uuid yet, it will be generated from the base64 you give us :)
    """

    image_uuid: Optional[str] = Field(None, description="The UUID for the image to be segmented", title="embedding")

    image_b64: Optional[str] = Field(None, description="The base64 encoded image", title="image")
    error_message: Optional[str] = Field(
        None,
        description="Details about any error that may have occurred",
        title="success",
    )

    text_prompt: Optional[List[float]] = Field(
        default=None,
        description="The text prompt for the thing you wanna segment in the image",
        title="text_prompt",
    )

    input_points: Optional[List[List[Union[float, int]]]] = Field(
        default=None,
        description="The json encoded points for the image",
        title="points",
    )
    input_labels: Optional[List[int]] = Field(
        default=None,
        description="The labels of the points. 1 for a positive point, 0 for a negative",
        title="labels",
    )
    input_boxes: Optional[Union[List[List[Union[int, float]]], List[Union[int, float]]]] = Field(
        default=None,
        description="The boxes for the image. For now, we only accept one",
        title="boxes",
    )

    image_shape: Optional[List[int]] = Field(
        default=None,
        description="The shape of the image (y_dim, x_dim)",
        title="image_shape",
    )

    masks: Optional[List[List[List[int]]]] = Field(
        default=None,
        description="The json encoded RLE masks",
        title="masks",
    )

    def deserialize(self) -> Optional[str]:
        """
        Deserialize the emebeddings response
        """
        return self.masks
