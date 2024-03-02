<div align="center">

# **👀 Vision [τ, τ]**
Giving 👀 to Bittensor with Decentralized subnet inference, at scale.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Bittensor](https://bittensor.com/whitepaper)  •  [Discord](https://discord.gg/dR865yTPaZ) • [Corcel](https://app.corcel.io/studio)
</div>


# Intro 📜
Subnet 19 is a Bittensor subnetwork, focused around inference at scale, with an initial focus on image generation & manipulation models.

### 👑 Highlights
- A ton of models supported - with arbitrary models to come 🙏🏻
- DPO technique integrated into scoring - miners compete head to head! 💀
- Custom trained XGBoost reward model to ensure miners can't game the system whilst allowing a variety of GPU types 📈
- Validators can sell their organic traffic in a completely decentralized way 💰

We have SDXL Turbo, Kandinsky, Inpainting, Scribble, Controlnets, Upscaling, Image-To-Image, Uncrop support, the list goes on!

## Installation
### [Miners](docs/miner_setup.md)

### [Validators](docs/validator_setup.md)


## Requirements

### Miners
GPU: 1x80 GB or 3x24gb or 2x48gb vram

You might be able to use 2x24gb vram or 1x48gb vram, but as models change this is subject to change

Storage: 200Gb

RAM: 32Gb

### Validators
GPU: 1x80 GB or 1x24GB or 1x48GB

Storage: 200Gb

Ram: 64Gb (Recommended)
### Tested Hardware
Setups tested for mining so far:

| Name  | CUDA Version | Ubuntu Version | Python Version | Works |
|-------|--------------|----------------|----------------|-------|
| H1OO  | 12.0 | 22.04 | 3.10.12 | ✅  |
| A100 | 11.8  | 22.04 | 3.10.12 | ✅ / ❌ (hit and miss) |
| RTX 4090 | 11.8  | 22.04 | 3.10.12 | ✅ |
| A6000* | 12.0   | 22.04 | 3.10.12 |✅ |
| A40 | 12.0   | 22.04 | 3.10.12 | ✅ |
| L40 | 12.0   | 22.04 | 3.10.12 | ❌ |
| A100 SXM | 11.8  | 22.04 | 3.10.12 | ❌|


Note: That's not to say you can't use other GPU's!

## Future of 19 + DSIS 🚀

The DSIS framework is committed to supporting a variety of models and workflows, optimizing inference. It primarily focuses on maintaining the highest throughput possible for production cases and APIs, while building a dynamic marketplace. The first phase focuses on a reward system proportional to the number of models run.

We already provide API access for high-scale inference, but we will do even more.

**Key future features include:**

-  **Subnet 19 for End Users**: Auto-scaling, fault tolerance, and reliability aspects will be catered to end users through Subnet 19.
-  **Simplified Validator Services**: Providing comprehensive services to validators with one-click validator launches and simple GUI's for managing access & payments
-  **Dynamic Resource Balance**: Allowing the running of an arbitrary number of models, at an enterprise scale
-  **Intersubnet Collaborations**: Maximizing value extraction from other subnets and enabling the use of bittensor in a plug-and-play fashion.
