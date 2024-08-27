# Small Tweaks For PaliGemma (Ongoing)

![PaliGemma Logo](./assets/paligemma_arch.png)

## Introduction

PaliGemma is a cutting-edge multimodal AI model that combines the power of language understanding with visual comprehension. Built on the latest advancements in machine learning, PaliGemma offers state-of-the-art performance in tasks involving both text and images. 


## Improvements

- **Flash Attention**: Implemented for both language and vision models, significantly improving computational efficiency speeding up the inference time.
- **Training**: Added Training loop as well as simple dataset support.

## TODO
- Implement SoViT-400m architecture, which is the shape-optimized version of the original SigLIP.
- Get pretrained model and perform finetune on user data.
## Acknowledgements

- Huggingface implementation can be found [here](https://github.com/hkproj/pytorch-paligemma/tree/main)

