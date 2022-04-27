import itertools
import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

exts = ["jpg", "jpeg", "png", "bmp"]


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
        if is_train:
            self.image_files = list(
                itertools.chain.from_iterable(
                    [glob(os.path.join(root, category, "train", "good", f"*.{ext}")) for ext in exts]
                )
            )
        else:
            self.image_files = list(
                itertools.chain.from_iterable(
                    [glob(os.path.join(root, category, "test", "*", f"*.{ext}")) for ext in exts]
                )
            )

        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        file_name = os.path.split(self.image_files[index])[1]
        if self.is_train:
            return image
        else:
            return image, file_name

    def __len__(self):
        return len(self.image_files)
