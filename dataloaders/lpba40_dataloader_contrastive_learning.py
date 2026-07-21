import os.path
from dataloaders.base_dataset import BaseDataset
from dataloaders.base_dataset import get_transform_contrastive_learning as get_transform_train
from dataloaders.base_dataset import get_transform_contrastive_learning_test as get_transform_test
from PIL import Image
import SimpleITK as sitk
import random
import numpy as np
import copy
import torch
from skimage import exposure



class LPBA40(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.image_path = self.opt.dataroot

        self.is_training = opt.phase == "train"
        self.is_testing = opt.phase == "test"

        # ~~~~~~~~~~~~~~~~~~~ paths ~~~~~~~~~~~~~~~~~~~
        if self.is_training:
            self.constrain = list(range(1, 189, 1))
        else:
            self.constrain = list(range(189, 208, 1))

        self.moving_path = []
        list_dirs = os.walk(self.image_path)
        for c in self.constrain:
            idx = f"{c:04d}"
            moving_file = os.path.join(self.image_path, f"OASIS_{idx}_0001.nii.gz")
            fixed_file = os.path.join(self.image_path, f"OASIS_{idx}_0000.nii.gz")
                        
            if os.path.exists(moving_file) and os.path.exists(fixed_file):
                self.moving_path.append(moving_file)
            else:
                print(f"Warning: missing pair for {idx}")

    
        self.moving_fixed = {
            moving: moving.replace("_0001.nii.gz", "_0000.nii.gz")
            for moving in self.moving_path
        }

        self.fixed_path = list(set(self.moving_fixed.values()))

        # ~~~~~~~~~~~~~~~~~~~ volume ~~~~~~~~~~~~~~~~~~~
        # Images
        self.fixed_img = {x: self.readVol(x) for x in self.fixed_path}

        # Transformation
        self.transform_train = get_transform_train()
        self.transform_test = get_transform_test()

    def name(self):
        return 'LPBA40_contrastive_learning'

    def __len__(self):
        return len(self.moving_path) if self.is_training else len(self.moving_path)

    def __getitem__(self, index):
        # get train or validation dataloaders
        if self.is_training:
            img_index = int(index % len(self.moving_path))
            

            moving_img = self.readVol(self.moving_path[img_index])
            fixed_img = self.readVol(self.moving_fixed[self.moving_path[img_index]])
            
            moving_atlas = np.round(self.readVol(
                self.moving_path[img_index].replace("imagesTr",
                                                    "masksTr"))).astype(np.uint8)
            fixed_atlas = np.round(self.readVol(
                self.moving_fixed[self.moving_path[img_index]].replace("imagesTr",
                                                                       "masksTr"))).astype(np.uint8)
            
            moving_img_pytorch, fixed_img_pytorch, moving_atlas_pytorch, fixed_atlas_pytorch = self.transform_train(
                [moving_img, fixed_img, moving_atlas, fixed_atlas])

            return {'A': moving_img_pytorch,
                    'A_atlas': moving_atlas_pytorch,
                    'A_paths': self.moving_path[img_index],
                    'B': fixed_img_pytorch,
                    'B_atlas': fixed_atlas_pytorch,
                    'B_paths': self.moving_fixed[self.moving_path[img_index]]}
        else:
            img_index = int(index % len(self.moving_path))

            moving_img = self.readVol(self.moving_path[img_index])
            fixed_img = self.readVol(self.moving_fixed[self.moving_path[img_index]])
           
            
            moving_atlas = np.round(self.readVol(self.moving_path[img_index].replace("imagesTr",
                                                                       "masksTr"))).astype(np.uint8)
            fixed_atlas = np.round(self.readVol(self.moving_fixed[self.moving_path[img_index]].replace("imagesTr",
                                                                       "masksTr"))).astype(np.uint8)

            
            moving_img_pytorch, fixed_img_pytorch, moving_atlas_pytorch, fixed_atlas_pytorch = self.transform_test(
                [moving_img, fixed_img, moving_atlas, fixed_atlas])

            return {'A': moving_img_pytorch,
                    'A_atlas': moving_atlas_pytorch,
                    'A_paths': self.moving_path[img_index],
                    'B': fixed_img_pytorch,
                    'B_atlas': fixed_atlas_pytorch,
                    'B_paths': self.moving_fixed[self.moving_path[img_index]]}

    def readVol(self, volpath):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(volpath))).swapaxes(0, 2)
    '''
    def whitening(self, image):
        """Not real Whitening. Just standardize image to 0-1."""
        image = image.astype(np.float32)

        return (np.clip(image, -1000., 238.) + 1000.) / (238 + 1000)
    '''
    
    def moving_whitening(self, image):
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        if max_val == min_val:
            return np.zeros_like(image)
        return (image - min_val) / (max_val - min_val)
    
    def fixed_whitening(self, image):
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        if max_val == min_val:
            return np.zeros_like(image)
        return (image - min_val) / (max_val - min_val)


def readVol(volpath):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(volpath))).swapaxes(0, 2)