# from fastapi.testclient import TestClient
# from validation.validator_checking_server.CHECKING_SERVER import app
# import pytest

# client = TestClient(app)

# def test_text_to_image():
#     response = client.post(
#         "/scoring/text-to-image",
#         json={
#             "body": {
#                 "text_prompts": [{"text": "whale"}],
#                 "height": 1024,
#                 "width": 1024,
#                 "cfg_scale": 1.0,
#                 "steps": 4,
#                 "seed": 128,
#                 "engine": "kandinsky-2.2"
#             },
#             "axon_response" : {
#                 "image_b64s" : ["base64_string"]
#             },
#             "response_time": 2.78
#         })

#     print(response.content)
#     assert response.status_code == 200
#     assert isinstance(response.json(), float)
