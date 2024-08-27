from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_length, image_token):
    return f"{image_token * image_seq_length}{bos_token}{prefix_prompt}\n"


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> Image:
    resized_image = image.resize(size, resample=resample, reducing_gap=reducing_gap)
    return resized_image


def rescale(image: np.ndarray, scale: float) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(np.float32)
    return rescaled_image


def normalize(image: np.ndarray, mean, std) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    normalized_image = (image - mean) / std
    return normalized_image


def process_images(images, size, resample, rescale_factor, image_mean, image_std):
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    images = [np.array(image) for image in images]
    images = [rescale(image=image, scale=rescale_factor) for image in images]
    images = [
        normalize(image=image, mean=image_mean, std=image_std) for image in images
    ]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens, image_size):
        self.tokenizer = tokenizer
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]

        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_boss_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longes",
        truncation: bool = True,
    ) -> dict:

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_length=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
