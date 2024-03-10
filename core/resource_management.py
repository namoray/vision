import asyncio
import contextvars
import gc
import threading
from collections.abc import Iterable
from functools import partial
from typing import NamedTuple, Optional, Dict, Any

import torch
import bittensor as bt
import clip
import datasets
import diskcache
import markovify
import yaml
from diffusers import (
    AutoPipelineForImage2Image,
    ControlNetModel,
    DDPMScheduler,
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
import time

from core import constants as cst
from core import kandinsky_utils, utils
from models import protocols

asyncio_lock: asyncio.Lock = asyncio.Lock()
threading_lock: threading.Lock = threading.Lock()

hotkey_name_var = contextvars.ContextVar("hotkey_name", default=None)


def set_hotkey_name(name: str):
    hotkey_name_var.set(name)


class ResourceConfig(NamedTuple):
    CLIP_DEVICE: Optional[str] = None
    SAM_DEVICE: Optional[str] = None
    SDXL_TURBO_DEVICE: Optional[str] = None
    KANDINSKY_DEVICE: Optional[str] = None
    SCRIBBLE_DEVICE: Optional[str] = None
    UPSCALE_DEVICE: Optional[str] = None
    SOTA_PROVIDER: Optional[str] = None
    SAFETY_CHECKERS_DEVICE: Optional[str] = None
    IS_VALIDATOR: bool = False


def get_hotkey_config_value(config: Dict[str, Any], key: str):
    if config.get(cst.SINGULAR_GPU) is not None:
        return config[cst.SINGULAR_GPU]
    else:
        return config.get(key)
class SingletonResourceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonResourceManager, cls).__new__(cls)
            cls._instance._loaded_resources = {}
            cls._instance._available_operations  = {}
        return cls._instance

    def __init__(self):
        self.resource_name_to_load_function = {
            # cst.MODEL_SAM: self.load_sam_resources,
            cst.MODEL_CLIP: self.load_clip_resources,
            cst.MODEL_SDXL_TURBO: self.load_sdxl_turbo_resources,
            cst.MODEL_KANDINSKY: self.load_kandinsky_resources,
            cst.MODEL_SCRIBBLE: self.load_scribble_resources,
            cst.MODEL_MARKOV: self.load_validator_resources,
            cst.MODEL_UPSCALE: self.load_upscale_resources,
            cst.MODEL_CACHE: self.load_cache,
            cst.IMAGE_SAFETY_CHECKERS: self.load_safety_checkers,
        }

    def load_config(self, config: Optional[ResourceConfig] = None):
        if config is None:
            hotkey_name = hotkey_name_var.get()
            bt.logging.info(f"Using hotkey: {hotkey_name} to load the config ⏳")
            yaml_config = yaml.safe_load(open(cst.CONFIG_FILEPATH))
            hotkey_config = yaml_config.get(hotkey_name, {})


            self._config = {
                cst.MODEL_CLIP: get_hotkey_config_value(hotkey_config, cst.CLIP_DEVICE_PARAM),
                cst.MODEL_SDXL_TURBO: get_hotkey_config_value(hotkey_config, cst.SDXL_TURBO_DEVICE_PARAM),
                cst.MODEL_KANDINSKY: get_hotkey_config_value(hotkey_config, cst.KANDINSKY_DEVICE_PARAM),
                cst.MODEL_SCRIBBLE: get_hotkey_config_value(hotkey_config, cst.SCRIBBLE_DEVICE_PARAM),
                cst.MODEL_UPSCALE: get_hotkey_config_value(hotkey_config, cst.UPSCALE_DEVICE_PARAM),
                cst.MODEL_SOTA: get_hotkey_config_value(hotkey_config, cst.SOTA_PROVIDER_PARAM),
                cst.IS_VALIDATOR: hotkey_config.get(cst.IS_VALIDATOR, False),
                cst.IMAGE_SAFETY_CHECKERS: get_hotkey_config_value(hotkey_config, cst.SAFETY_CHECKERS_PARAM),
                # cst.MODEL_SAM: get_hotkey_config_value(hotkey_config, cst.SAM_DEVICE),
            }
        else:
            self._config = {
                cst.MODEL_CLIP: config.CLIP_DEVICE,
                cst.MODEL_SDXL_TURBO: config.SDXL_TURBO_DEVICE,
                cst.MODEL_KANDINSKY: config.KANDINSKY_DEVICE,
                cst.MODEL_SCRIBBLE: config.SCRIBBLE_DEVICE,
                cst.MODEL_UPSCALE: config.UPSCALE_DEVICE,
                cst.MODEL_SOTA: config.SOTA_PROVIDER,
                cst.IS_VALIDATOR: config.IS_VALIDATOR,
                cst.IMAGE_SAFETY_CHECKERS: config.SAFETY_CHECKERS_DEVICE,
                # cst.MODEL_SAM: config.SAM_DEVICE,
            }

    def load_all_resources(self):
        self.load_safety_checkers()

        # if self._config[cst.MODEL_SAM] is not None:
        #     self.load_resource(cst.MODEL_SAM)
        if self._config[cst.IS_VALIDATOR]:
            self.load_validator_resources()

        if self._config[cst.MODEL_CLIP] is not None:
            self.load_resource(cst.MODEL_CLIP)

        if self._config[cst.MODEL_SDXL_TURBO] is not None:
            self.load_resource(cst.MODEL_SDXL_TURBO)

        if self._config[cst.MODEL_KANDINSKY] is not None:
            self.load_resource(cst.MODEL_KANDINSKY)

        if self._config[cst.MODEL_SCRIBBLE] is not None:
            self.load_resource(cst.MODEL_SCRIBBLE)

        if self._config[cst.MODEL_UPSCALE] is not None:
            self.load_resource(cst.MODEL_UPSCALE)


        self.load_resource(cst.MODEL_CACHE)

    def load_safety_checkers(self):
        safety_checker_device = self._config.get(cst.IMAGE_SAFETY_CHECKERS, None)
        if safety_checker_device is None:
            raise ValueError("You MUST provide a device to run the safety checkers on")

        print("safety checker device:", safety_checker_device)

        safety_pipe = StableDiffusionPipeline.from_pretrained(
            cst.DREAMSHAPER_PIPELINE_REPO, torch_dtype=torch.bfloat16, cache_dir=cst.MODELS_CACHE
        )
        safety_pipe.to(safety_checker_device)
        safety_pipe.safety_checker.forward = partial(utils.forward_inspect, self=safety_pipe.safety_checker)
        self._loaded_resources[cst.IMAGE_SAFETY_CHECKERS] = (safety_pipe.feature_extractor, safety_pipe.safety_checker)

    def load_upscale_resources(self):
        upscale_device = self._config.get(cst.MODEL_UPSCALE, None)

        upscale_model = utils.RealESRGANClass(upscale_device, scale=2)
        upscale_model.load_weights('weights/RealESRGAN_x2.pth', download=True)

        self._loaded_resources[cst.MODEL_UPSCALE] = upscale_model
        self._update_available_operations(protocols.Upscale.__name__, True)


    def load_sam_resources(self):
        # TODO: Re-enable
        return
        # sam_model = sam.sam_model_registry[cst.MODEL_TYPE](checkpoint=cst.CHECKPOINT_PATH)
        # sam_device = self._config.get(cst.MODEL_SAM, cst.DEVICE_DEFAULT)
        # sam_model.to(sam_device)
        # sam_predictor = sam.SamPredictor(sam_model)
        # self._loaded_resources[cst.MODEL_SAM] = (sam_model, sam_predictor)
        # self._update_available_operations(protocols.Segment.__name__, True)

    def load_clip_resources(self):
        clip_device = self._config.get(cst.MODEL_CLIP, cst.DEVICE_DEFAULT)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)
        self._loaded_resources["clip"] = (clip_model, clip_preprocess)
        self._update_available_operations(protocols.ClipEmbeddings.__name__, True)


    def load_sdxl_turbo_resources(self):
        sdxl_turbo_device = self._config.get(cst.MODEL_SDXL_TURBO, cst.DEVICE_DEFAULT)
        sdxl_turbo_base_pipe = StableDiffusionXLPipeline.from_single_file(cst.DREAMSHAPER_XL_LOCAL_FILE, torch_dtype=torch.bfloat16, use_safetensors=True).to(sdxl_turbo_device)
        config = sdxl_turbo_base_pipe.scheduler.config
        scheduler = DDPMScheduler.from_config(config)
        sdxl_turbo_base_pipe.scheduler = scheduler
        sdxl_img2img_pipe = AutoPipelineForImage2Image.from_pipe(sdxl_turbo_base_pipe).to(sdxl_turbo_device)

        self._loaded_resources[cst.MODEL_SDXL_TURBO] = (sdxl_turbo_base_pipe, sdxl_img2img_pipe)
        self._update_available_operations(protocols.TextToImage.__name__, True)
        self._update_available_operations(protocols.ImageToImage.__name__, True)

    def load_kandinsky_resources(self):
        kandinsky_device = self._config.get(cst.MODEL_KANDINSKY, cst.DEVICE_DEFAULT)
        prior = KandinskyV22PriorPipeline.from_pretrained(
            cst.KANDINSKY_2_2_PRIOR_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=cst.MODELS_CACHE
        ).to(kandinsky_device)

        text2img = KandinskyV22Pipeline.from_pretrained(
            cst.KANDINSKY_2_2_DECODER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=cst.MODELS_CACHE,
            use_safetensors=True
        ).to(kandinsky_device)


        inpaint = KandinskyV22InpaintPipeline.from_pretrained(
            cst.KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=cst.MODELS_CACHE,
            use_safetensors=True
        ).to(kandinsky_device)

        kandinsky_pipe = kandinsky_utils.KandinskyPipe_2_2(
            prior=prior,
            text2img=text2img,
            inpaint=inpaint,
        )

        self._loaded_resources[cst.MODEL_KANDINSKY] = kandinsky_pipe
        self._update_available_operations(protocols.TextToImage.__name__, True)
        self._update_available_operations(protocols.ImageToImage.__name__, True)
        self._update_available_operations(protocols.Inpaint.__name__, True)

    def load_scribble_resources(self):
        scribble_device = self._config.get(cst.MODEL_SCRIBBLE, cst.DEVICE_DEFAULT)
        scribble_scheduler = DDPMScheduler.from_pretrained(cst.DREAMSHAPER_PIPELINE_REPO, subfolder="scheduler", torch_dtype=torch.bfloat16, cache_dir=cst.MODELS_CACHE)
        scribble_controlnet = ControlNetModel.from_pretrained(cst.CONTROL_MODEL_REPO, torch_dtype=torch.bfloat16, cache_dir=cst.MODELS_CACHE)
        scribble_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            cst.DREAMSHAPER_PIPELINE_REPO,
            scheduler=scribble_scheduler,
            controlnet=scribble_controlnet,
            torch_dtype=torch.bfloat16,
            cache_dir=cst.MODELS_CACHE
        ).to(scribble_device)

        self._loaded_resources[cst.MODEL_SCRIBBLE] = scribble_pipeline
        self._update_available_operations(protocols.Scribble.__name__, True)

    def load_validator_resources(self):
        dataset = datasets.load_dataset(cst.DATASET_REPO)
        text = [i["query"] for i in dataset["train"]]
        markov_text_generation_model = markovify.Text(" ".join(text))

        self._loaded_resources[cst.MODEL_MARKOV] = markov_text_generation_model

    def load_cache(self):
        cache = diskcache.Cache(cst.CACHE_PATH, size_limit=cst.CACHE_SIZE)
        self._loaded_resources[cst.MODEL_CACHE] = cache

    def load_resource(self, resource_name: str):
        if resource_name not in self._loaded_resources:
            if resource_name in self.resource_name_to_load_function:
                load_function = self.resource_name_to_load_function[resource_name]
                load_function()
        else:
            resources = self._loaded_resources[resource_name]
            try:
                device = self._config[resource_name]
            except KeyError:
                # No device for this resource, e.g. cache
                return

            if isinstance(resources, Iterable):
                for resource in resources:
                    try:
                        resource.to(device)
                    except AttributeError:
                        pass
            else:
                try:
                    resources.to(device)
                except AttributeError:
                    pass

    def get_resource(self, resource_name: str):
        self.load_resource(resource_name)
        model = self._loaded_resources[resource_name]
        return model

    def unload_resource(self, resource_name: str) -> None:
        if resource_name in self._loaded_resources:
            resources = self._loaded_resources.pop(resource_name)

            if isinstance(resources, Iterable):
                for resource in resources:
                    del resource
            else:
                del resources

            gc.collect()
            torch.cuda.empty_cache()
            self._update_available_operations(resource_name, False)

    def move_resource_to_cpu(self, resource_name: str) -> None:
        if resource_name in self._loaded_resources:
            resources = self._loaded_resources[resource_name]

            if isinstance(resources, Iterable):
                for resource in resources:
                    try:
                        resource.to("cpu")
                    except AttributeError:
                        ...
            else:
                try:
                    resources.to("cpu")
                except AttributeError:
                    ...

            torch.cuda.empty_cache()
            self._update_available_operations(resource_name, False)

    def unload_all_models(self):
        """
        Unloads all GPU models, except the safety checkers
        """

        for resource_name in list(self._loaded_resources.keys()):
            if resource_name not in [
                cst.MODEL_CACHE,
                cst.MODEL_MARKOV,
                cst.PROMPT_SAFETY_CHECKERS,
                cst.IMAGE_SAFETY_CHECKERS,
            ]:
                self.unload_resource(resource_name)

    def move_all_models_to_cpu(self):
        resource_times = {}
        for resource_name in list(self._loaded_resources.keys()):
            if resource_name not in [
                cst.MODEL_CACHE,
                cst.MODEL_MARKOV,
                cst.PROMPT_SAFETY_CHECKERS,
                cst.IMAGE_SAFETY_CHECKERS,
            ]:
                start_time = time.time()
                self.move_resource_to_cpu(resource_name)
                end_time = time.time()
                resource_times[resource_name] = end_time - start_time
                
        log_message = "\n".join(f"Moving resource {name} to CPU took {time:.2f} seconds" for name, time in resource_times.items())
        bt.logging.info(log_message)

    def get_available_operations(self):
        return self._available_operations

    def _update_available_operations(self, protocol_name: str, status: bool) -> None:
        self._available_operations[protocol_name] = status

    @classmethod
    def reset(cls):
        cls._instance = None
        cls.__new__(cls)
