from typing import Optional
from pydantic import BaseModel


class TextPrompt(BaseModel):
    text: str
    weight: Optional[float]
