# data/augmentations.py
import random
import torch

class SeldAugmentor:
    def __init__(self, img_h=180, img_w=360, n_acoustic=4):
        self.img_h, self.img_w, self.n_acoustic = img_h, img_w, n_acoustic
    def __call__(self, image: torch.Tensor, target: dict) -> tuple:
        if random.random() < 0.5:
            image[1:] = -image[1:]
            if "vmap" in target: target["vmap"] = target["vmap"].flip(-1)
        return image, target