
from typing import Dict

SCORING_CHANCE = 0.2
BOTTOM_PERCENTAGE_OF_MINERS_TO_IGNORE = 0.1

# This is to reward to best performing miners for how many queries they're doing, to offset their costs
SCORE_DISCOUNT_OFFSET = 0.01

SYNTHETIC_POST_URL_BASE = "synthetic"

MAX_INTERNAL_SERVER_ERRORS = 2

MINIMUM_ACTIVE_MINERS_TO_APPLY_TIERS = 20


NETUID = 51
NETWORK = "test"





OPERATION_TIMEOUTS: Dict[str, float] = {
    "AvailableOperations": 12,
    "ClipEmbeddings": 4,
    "TextToImage": 15,
    "ImageToImage": 15,
    "Upscale": 10,
    "Inpaint": 15,
    "Scribble": 12,
    # "Segment": 15,
}

# This is to stop people claiming operations that don't exist
ALLOWED_USEFUL_OPERATIONS = [
    "TextToImage",
    "ImageToImage",
    "Upscale",
    "Inpaint",
    "Scribble",
    "ClipEmbeddings",
    # "Segment",
]

# FOR PHASE 1 - where synthetic only validators may have a distribution different to organic ones
AVAILABLE_OPERATIONS_MULTIPLIER = {0: 0, 1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.9, 6: 1, 7: 1.1, 8: 1.2}


OPERATIONS_TO_SCORE_SYNTHETICALLY = ALLOWED_USEFUL_OPERATIONS


MAX_INTERNAL_SERVER_ERRORS = 3
STATUS_OK = 200
STATUS_INTERNAL_SERVER_ERROR = 500
SCORE_QUERY_PROBABILITY = 0.05
NUMBER_OF_SECONDARY_AXONS_TO_COMPARE_WHEN_SCORING = 1  # head - head comparison
BONUS_FOR_WINNING_MINER = 0.1

CHANCE_TO_CHECK_OUTPUT_WHEN_IMAGES_FROM_MINERS_WERE_SIMILAR = 0.1
MINIMUM_SIMILARITY_WITH_VALIDATOR_RESULT = 0.1
SIMILARITY_BETWEEN_MINER_RESPONSES_THRESHOLD = (
    0.1  # If Miners have responses which are closer than this in similarity, assume they are both correct
)
