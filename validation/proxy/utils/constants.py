from typing import Dict

TESTNET = "testnet"


# This is to reward to best performing miners for how many queries they're doing, to offset their costs


NETUID = 19
NETWORK = "finney"
MAX_RESULTS_TO_SCORE_FOR_TASK = 100
MINIMUM_TASKS_TO_START_SCORING = 10

MIN_SECONDS_BETWEEN_SYNTHETICALLY_SCORING = 5

# NOTE: Make these more granular, based on the prediction of the response time perhaps?
OPERATION_TIMEOUTS: Dict[str, float] = {
    "AvailableTasksOperation": 30,
    "Capacity": 10,
    "ClipEmbeddings": 6,
    "TextToImage": 15,
    "ImageToImage": 15,
    "Upscale": 15,
    "Inpaint": 20,
    "Scribble": 20,
    "Avatar": 50,
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
