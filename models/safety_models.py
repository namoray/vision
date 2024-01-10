from typing import List

from pydantic import BaseModel


class CheckImageRequest(BaseModel):
    image_b64s: List[str]
