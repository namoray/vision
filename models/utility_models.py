import enum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel
import bittensor as bt


class UIDinfo(BaseModel) :


    class Config:
        arbitrary_types_allowed = True


    uid: int
    hotkey: str
    available_operations: List[str] = []
    axon: bt.chain_data.AxonInfo
    is_low_incentive: bool = False
    incentive: float = 0.0


    total_organic_score: float = 0.0
    organic_request_count: int = 0

    total_synthetic_score: float = 0.0
    synthetic_request_count: int = 0


    @property
    def average_score(self) -> float:
        """You get the minimum of the scores, to prevent attack vectors.

        If you didn't get any requests of one type, then I will just use the other."""

        if self.organic_request_count == 0 and self.synthetic_request_count == 0:
            return 0.0

        if self.organic_request_count == 0:
            return self.total_synthetic_score / self.synthetic_request_count

        elif self.synthetic_request_count == 0:
            return self.total_organic_score / self.organic_request_count


        average_organic_score = self.total_organic_score / max(1, self.organic_request_count)
        average_synthetic_score = self.total_synthetic_score / max(1, self.synthetic_request_count)
        return min(average_organic_score, average_synthetic_score)

    def add_score(self, score: float, synthetic: bool = False, count: Optional[int] = 1) -> None:
        if synthetic:
            self.total_synthetic_score += score * count
            self.synthetic_request_count += count
        else:
            self.total_organic_score += score * count
            self.organic_request_count += count

    def reset_scores(self) -> None:
        self.total_organic_score = 0
        self.organic_request_count = 0

        self.total_synthetic_score = 0
        self.synthetic_request_count = 0




class QueryResult(BaseModel):
    formatted_response: Optional[BaseModel]
    axon_uid: Optional[int]
    response_time: Optional[float]
    error_message: Optional[str]
    failed_axon_uids: List[int] = []


class OperationDistribution(BaseModel):
    available_axons: List[int]
    probabilities: List[float]
    score_discounts: Dict[int, float]

    def get_order_of_axons_to_query(self) -> List[int]:
        z = -np.log(-np.log(np.random.uniform(0, 1, len(self.available_axons))))
        scores = np.log(self.probabilities) + z
        return [self.available_axons[i] for i in np.argsort(-scores)]


class MaskSourceEnum(str, enum.Enum):
    MASK_IMAGE_WHITE = "MASK_IMAGE_WHITE"
    MASK_IMAGE_BLACK = "MASK_IMAGE_BLACK"
    INIT_IMAGE_ALPHA = "INIT_IMAGE_ALPHA"


class EngineEnum(str, enum.Enum):
    SDXL_TURBO = "stable-diffusion-xl-turbo"
    SDXL_1_5 = "stable-diffusion-1.5"
    KANDINSKY_22 = "kandinsky-2.2"

class ImageHashes(BaseModel):
    average_hash: str = ""
    perceptual_hash: str = ""
    difference_hash: str = ""
    color_hash: str = ""
