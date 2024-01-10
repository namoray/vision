
import bittensor as bt
import numpy as np
import torch

from core import constants as cst
from core import resource_management
from models import base_models
from operation_logic import utils


async def segment_logic(body: base_models.SegmentIncoming) -> base_models.SegmentOutgoing:
    threading_lock = resource_management.threading_lock
    sam_model, sam_predictor = resource_management.SingletonResourceManager().get_resource(cst.MODEL_SAM)
    output = base_models.SegmentOutgoing()

    bt.logging.debug("Gonna generate some masks like the good little miner I am")

    if body.image_b64 is None:  # and body.image_uuid is None and
        output.error_message = "❌ You must supply an image or UUID of the already stored image"
        bt.logging.warning(f"USER ERROR: {output.error_message}, body: {body}")
        return output

    if body.input_points is None and body.input_boxes is None and body.input_labels is None:
        output.error_message = "❌ No input points, boxes or labels"  # , just gonna store the image
        bt.logging.warning(f" USER ERROR: {output.error_message}")
        return output

    image_cv2 = utils.convert_b64_to_cv2_img(body.image_b64)
    with threading_lock:
        sam_predictor.set_image(image_cv2)
        if body.input_boxes is None or len(body.input_boxes) == 0 or isinstance(body.input_boxes[0], int) or len(body.input_boxes) == 1:
            input_points = np.array(body.input_points) if body.input_points else None
            input_labels = np.array(body.input_labels) if body.input_labels else None
            input_boxes = np.array(body.input_boxes).squeeze() if body.input_boxes else None

            all_masks, scores, _ = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_boxes,
                multimask_output=True,
            )

        else:
            input_boxes_tensor = torch.tensor(body.input_boxes, device=sam_predictor.device)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes_tensor, image_cv2.shape[:2])
            all_masks, scores, logits = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=True,
            )

            all_masks = all_masks.cpu().numpy()
            scores = scores.cpu().numpy()

        if len(all_masks.shape) == 4:
            best_options_indices = np.argmax(scores, axis=1)
            best_masks = all_masks[np.arange(all_masks.shape[0]), best_options_indices, :, :]

        else:
            best_score = np.argmax(scores)
            best_masks = [all_masks[best_score, :, :]]

        encoded_masks = utils.rle_encode_masks(best_masks)
        output.masks = encoded_masks
        output.image_shape = list(image_cv2.shape)[:2]

        if len(encoded_masks) > 0:
            bt.logging.info(f"✅ Generated {len(output.masks)} mask(s), go me")

    return output
