import numpy as np
from PIL import Image
import cv2

from cloth_segmentation import ClothSegmentor
from fill_anything import Filler


class Inferencer:

    def __init__(self, device):
        self.device = device
        self.segmentor = ClothSegmentor(device=self.device)
        self.filler = Filler(device=self.device)


    def inference(self, image_path, text_prompt):
        img = Image.open(image_path)
        img = np.array(img)
        mask = self.segmentor.generate_mask(image_path)
        np.savetxt('mask.txt', mask, fmt='%f')
        filled_img = self.filler.fill_img_with_sd(img, mask, text_prompt)
        return filled_img

image_path = './assets/03615_00.jpg'
text_prompt = "a dragon is flying around"

inferencer = Inferencer('cpu')
filled_img = inferencer.inference(image_path, text_prompt)
cv2.imwrite('./result/filled_im.jpg', filled_img)