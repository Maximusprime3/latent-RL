import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
#from torchvision.datasets import CelebA
#import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) # relative path to directory + filename        
        #print('AAA', self.data_dir)
        self.transforms = transform
        
        #cannot be torch needs to be string
        #imgs = torch.load(self.data_dir)

        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpeg'])
        print('MyDataset: ', len(imgs), ' images found in', self.data_dir)
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
        
        pass
    
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        #I added to normalize specific to my dataset mountaincar
        self.mean = (252.9872, 252.9880, 252.8330)
        self.std  = (17.3592, 17.3520, 18.5265)

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
  
#        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                              transforms.CenterCrop(148),
#                                              transforms.Resize(self.patch_size),
#                                              transforms.ToTensor(),])
#        
#        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                            transforms.CenterCrop(148),
#                                            transforms.Resize(self.patch_size),
#                                            transforms.ToTensor(),])
#        
#        self.train_dataset = MyCelebA(
#            self.data_dir,
#            split='train',
#            transform=train_transforms,
#            download=False,
#        )
#        
#        # Replace CelebA with your dataset
#        self.val_dataset = MyCelebA(
#            self.data_dir,
#            split='test',
#            transform=val_transforms,
#            download=False,
#        )

#       ===============================================================

# My own data setup Need the mean and std of dataset here in transforms.Normalize!
#       ===============================================================
         train_transforms = transforms.Compose([transforms.ToTensor(),
                                               #transforms.RandomHorizontalFlip(),
                                               transforms.Resize((self.patch_size, self.patch_size)),
                                               #transforms.Grayscale(),
                                               #only do this when you got the mean and std of the dataset beforehand
                                               #transforms.Normalize(self.mean, self.std), #need to figure this out before as it is dataset specific 
                                               #NormalizeImages(),
                                               AddGaussianNoise(0., 0.1),
                                               ClipValues(0., 255.),
                                               ])
        
         val_transforms = transforms.Compose([transforms.ToTensor(),
                                             #transforms.RandomHorizontalFlip(),
                                             transforms.Resize((self.patch_size, self.patch_size)),
                                             #transforms.Grayscale(),
                                            #only do this when you got the mean and std of the dataset beforehand
                                             #transforms.Normalize(self.mean, self.std),
                                             #NormalizeImages(),
                                             AddGaussianNoise(0., 0.1),
                                             ClipValues(0., 255.),
                                             ])

         self.train_dataset = MyDataset(
             self.data_dir,
             split='train',
             transform=train_transforms
             
         )
         #print('train_dataset: ', len(self.train_dataset))
      
        
         self.val_dataset = MyDataset(
             self.data_dir,
             split='val',
             transform=val_transforms
             
         )
         #print('val_dataset: ', len(self.val_dataset))

#       ===============================================================


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

#for data transforms   
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class NormalizeImages(object):
    def __init__(self, max_value=None, min_value=None):
        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, images):
        # Convert to floating point type
        images = images.float()

        # Compute maximum and minimum values if not provided
        if self.max_value is None or self.min_value is None:
            self.max_value = torch.max(images)
            self.min_value = torch.min(images)

        # Normalize the images
        normalized_images = (images - self.min_value) / (self.max_value - self.min_value)

        return normalized_images

class ClipValues(object):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor):
        return torch.clamp(tensor, self.min_value, self.max_value)