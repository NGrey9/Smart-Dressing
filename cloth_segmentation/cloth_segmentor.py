import numpy as np
import os
import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from PIL import Image

from network import U2NET

class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"



class ClothSegmentor:
    def __init__(self, device):
        self.device = device
        self.model = self.load_from_path()
        self.palette = self.get_palette(4)

    def load_from_path(self, checkpoint_path='./models/cloth_segmentation/cloth_segm.pth'):
        if not os.path.exists(checkpoint_path):
            print("Không có checkpoint trong thư mục models")
            return None
        net = U2NET(in_ch=3, out_ch=4)
        model_state_dict = torch.load(checkpoint_path, map_location=torch.device(self.device))
        new_state_dict = OrderedDict()

        for k, v in model_state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)
        net.to(self.device)
        print("----checkpoints loaded from path: {}----".format(checkpoint_path))
        return net

    def get_palette(self, num_cls):
        """ Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette
    

    def apply_transform(self, img):
        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(0.5, 0.5)]
        transform_rgb = transforms.Compose(transforms_list)
        return transform_rgb(img)
    
        
    def generate_mask(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_size = img.size
        img = img.resize((768, 768), Image.BICUBIC)
        image_tensor = self.apply_transform(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        
        os.makedirs("./result/alpha", exist_ok=True)
        os.makedirs("./result/cloth_seg", exist_ok=True)

        with torch.no_grad():
            output_tensor = self.model(image_tensor.to(self.device))
            output_tensor = F.log_softmax(output_tensor[0], dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_arr = output_tensor.cpu().numpy()
            
        classes_to_save = []
        # Check which classes are present in the image
        for cls in range(1, 4):  # Exclude background class (0)
            if np.any(output_arr == cls):
                classes_to_save.append(cls)

        # Save alpha masks
        for cls in classes_to_save:
            alpha_mask = (output_arr == cls).astype(np.uint8) * 255
            alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
            alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
            alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
            alpha_mask_img.save(os.path.join("./result/alpha/", f'{cls}.png'))

        # Save final cloth segmentations
        cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
        cloth_seg.putpalette(self.palette)
        cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
        cloth_seg.save(os.path.join("./result/cloth_seg", 'final_seg.png'))


input_image_path = './assets/model-873675_1280.jpg'

segmentor = ClothSegmentor(device="cpu")
segmentor.generate_mask(input_image_path)


