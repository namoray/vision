import dataclasses
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import DDPMScheduler, KandinskyV22InpaintPipeline, KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.utils.outputs import BaseOutput
from PIL import Image

from core import constants as cst


@dataclasses.dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[Image.Image], np.ndarray]


num_outputs = 1


class KandinskyPipe_2_2:
    def __init__(
        self,
        prior: KandinskyV22PriorPipeline,
        text2img: KandinskyV22Pipeline,
        inpaint: KandinskyV22InpaintPipeline,
    ):
        self.prior = prior
        self.text2img = text2img
        self.inpaint = inpaint

        self.text2img.scheduler = DDPMScheduler.from_config(self.text2img.scheduler.config)
        self.inpaint.scheduler = DDPMScheduler.from_config(self.inpaint.scheduler.config)

    def to(self, device):
        self.prior.to(device)
        self.text2img.to(device)
        self.inpaint.to(device)

    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        generator: torch.Generator = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        strength: Optional[float] = None,
    ) -> ImagePipelineOutput:
        if generator is None:
            raise ValueError("No generator provided when calling kandinsky pipeline! Something has gone wrong")

        if image is None and mask_image is None:
            img_emb = self.prior(
                prompt=prompt,
                num_inference_steps=cst.PRIOR_STEPS,
                guidance_scale=cst.PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            neg_emb = self.prior(
                prompt=negative_prompt,
                num_inference_steps=cst.PRIOR_STEPS,
                guidance_scale=cst.PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            pipe_output = self.text2img(
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=neg_emb.image_embeds,
            )
            return pipe_output

        elif image is not None and mask_image is None:
            images_and_texts = [prompt, image]
            weights = [strength, 1 - strength]
            prior_out = self.prior.interpolate(
                images_and_texts,
                weights,
                negative_prompt=negative_prompt,
                num_inference_steps=cst.PRIOR_STEPS,
                guidance_scale=cst.PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            output_images = self.text2img(
                **prior_out,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            return output_images

        elif image is not None and mask_image is not None:
            img_emb = self.prior(
                prompt=prompt,
                num_inference_steps=cst.PRIOR_STEPS,
                guidance_scale=cst.PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            neg_emb = self.prior(
                prompt=negative_prompt,
                num_inference_steps=cst.PRIOR_STEPS,
                guidance_scale=cst.PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            pipe_output = self.inpaint(
                image=[image] * 1,
                mask_image=[mask_image] * 1,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=neg_emb.image_embeds,
                width=image.width,
                height=image.height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return pipe_output
