import base64
import os
import urllib.request
from typing import Optional


def get_testing_image(image_description: str) -> str:
    dir_path = "tests/test_images"
    local_file_path = f"{dir_path}/{image_description}.webp"
    os.makedirs(dir_path, exist_ok=True)

    if not os.path.isfile(local_file_path):
        url = f"https://storage.googleapis.com/testing-image/{image_description}.webp"
        urllib.request.urlretrieve(url, local_file_path)

    with open(local_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    return encoded_string


def save_image(
    model_name: str,
    cfg_scale: float,
    steps: int,
    image_b64: str,
    action: str = "image_to_image",
    image_strength: Optional[float] = 0.7,
) -> None:
    dir_path = "tests/test_images"
    os.makedirs(dir_path, exist_ok=True)

    if image_strength:
        filename = f"{action}_{model_name}_{cfg_scale}_{steps}_{image_strength}.webp"
    else:
        filename = f"{action}_{model_name}_{cfg_scale}_{steps}.webp"

    # decode the base64 image and write to file
    with open(os.path.join(dir_path, filename), "wb") as f:
        f.write(base64.b64decode(image_b64))
