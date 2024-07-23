import cv2
import sys
sys.path.append("./fill_anything/")
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from utils import (crop_for_filling_pre,crop_for_filling_post,
                   resize_and_pad, recover_size)

class Filler:

    def __init__(self, device):
        self.device = device
        self.pipe = self.load_from_path()

    def load_from_path(self, model_name="stabilityai/stable-diffusion-2-inpainting"):
        return StableDiffusionInpaintPipeline.from_pretrained(model_name,
                                                              torch_dtype=torch.float32).to(self.device)
    
    def fill_img_with_sd(self, img, mask, text_prompt):
        img_crop, mask_crop = crop_for_filling_pre(img, mask)
        img_crop_filled = self.pipe(
            prompt=text_prompt,
            image=Image.fromarray(img_crop),
            mask_image=Image.fromarray(mask_crop)
        ).images[0]

        img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
        return img_filled
    
    def replace_img_with_sd(self, img, mask, text_prompt, step:int=50):
        img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
        img_padded = self.pipe(
            prompt=text_prompt,
            image=Image.fromarray(img_padded),
            mask_image=Image.fromarray(255 - mask_padded),
            num_inference_steps=step,
        ).images[0]
        height, width, _ = img.shape
        img_resized, mask_resized = recover_size(
            np.array(img_padded), mask_padded, (height, width), padding_factors)
        mask_resized = np.expand_dims(mask_resized, -1) / 255
        img_resized = img_resized * (1-mask_resized) + img * mask_resized
        return img_resized






