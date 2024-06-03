from typing import Dict

SCORING_CHANCE = 0.2
BOTTOM_PERCENTAGE_OF_MINERS_TO_IGNORE = 0.1

# This is to reward to best performing miners for how many queries they're doing, to offset their costs
SCORE_DISCOUNT_OFFSET = 0.01

SYNTHETIC_POST_URL_BASE = "synthetic"

MAX_INTERNAL_SERVER_ERRORS = 2

MINIMUM_ACTIVE_MINERS_TO_APPLY_TIERS = 20


NETUID = 19
NETWORK = "finney"
MAX_RESULTS_TO_SCORE_FOR_TASK = 50
MINIMUM_TASKS_TO_START_SCORING = 40

# TODO: Make these more granular, based on the prediction of the response time perhaps?
OPERATION_TIMEOUTS: Dict[str, float] = {
    "AvailableTasksOperation": 30,
    "ClipEmbeddings": 6,
    "TextToImage": 15,
    "ImageToImage": 15,
    "Upscale": 15,
    "Inpaint": 20,
    "Scribble": 20,
    "Avatar": 50,
    "Sota": 180,
    "Chat": 60,
}

# FOR PHASE 1 - where synthetic only validators may have a distribution different to organic ones
AVAILABLE_TASKS_MULTIPLIER = {
    0: 0,
    1: 0.4,
    2: 0.4,
    3: 0.5,
    4: 0.5,
    5: 0.5,
    6: 0.6,
    7: 0.7,
    8: 0.7,
    9: 0.8,
    10: 1.0,
    11: 1.1,
    12: 1.4,
    13: 1.5,
}


MAX_INTERNAL_SERVER_ERRORS = 3
STATUS_OK = 200
STATUS_INTERNAL_SERVER_ERROR = 500


SCORE_QUERY_PROBABILITY = 0.1
BONUS_FOR_WINNING_MINER = 0.25
FAILED_RESPONSE_SCORE = 0.25

CHANCE_TO_CHECK_OUTPUT_WHEN_IMAGES_FROM_MINERS_WERE_SIMILAR = 0.02

MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING = 2
