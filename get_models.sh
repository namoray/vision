#!/bin/bash
if [ -f dreamshaper_xl.safetensors ]; then
    echo "Dreamshaper model exists, no need to download"
else
    wget -O dreamshaper_xl.safetensors https://civitai.com/api/download/models/333449?type=Model&format=SafeTensor&size=full&fp=fp16
    exit 0
fi


# if [ -f stable-diffusion-safety-checker/pytorch_model.bin ]; then
#     echo "Safety checker model already exists, no need to download"
# else
#     wget -O pytorch_model.bin https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/cb41f3a270d63d454d385fc2e4f571c487c253c5/pytorch_model.bin
#     mv pytorch_model.bin stable-diffusion-safety-checker/
# fi

# SAM_FILE=sam_vit_l_0b3195.pth
# URL=https://dl.fbaipublicfiles.com/segment_anything/$SAM_FILE
# if [ -f "$SAM_FILE" ]; then
#     echo "$SAM_FILE exists, no need to download"
# else
#     echo "$SAM_FILE does not exist, starting download."
#     curl -O $URL
# fi

# https://huggingface.co/SG161222/RealVisXL_V3.0_Turbo
