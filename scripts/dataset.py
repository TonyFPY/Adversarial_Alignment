import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
    
class ClickMe(Dataset):
    def __init__(self, file_paths, model_name='resnet18.tv_in1k', imagenet2superclass_mapper=None, need_normalize=True, is_training=False):
        super(Dataset).__init__()
        self.file_paths = file_paths
        self.is_training = is_training
        self.need_normalize = need_normalize
        self.in2sc_mapper = imagenet2superclass_mapper

        data_config = resolve_model_data_config(model_name)
        timm_transforms_normalize = create_transform(**data_config).transforms[-1]
        timm_transforms_resize = create_transform(**data_config).transforms[0]
        img_size = data_config['input_size'][-1]
 
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.0)),  # Randomly crop and resize the image tensor
                transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a probability of 0.5
                transforms.RandomRotation(degrees=(-15, 15)),
            ]),
            'eval': transforms.Compose([
                timm_transforms_resize,
                transforms.CenterCrop(img_size)
            ]),   
            'norm': transforms.Compose([
                timm_transforms_normalize,
            ]),   
        }

    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        img, hmp, label = data['image'], data['heatmap'], data['label']
        
        img = img.to(torch.float32) / 255.0 # uint8 -> float32
        hmp = hmp.to(torch.float32) / 255.0 # uint8 -> float32
        if self.in2sc_mapper:
            label = torch.tensor(self.in2sc_mapper[label.item()][2]) # key = 0; val = ('n01440764', 'tench', 9, 'other')
        else:
            label = label.to(torch.int64)       # int32 -> int64
        label = torch.squeeze(label)        # [batch_size, 1] -> [batch_size]
        
        stacked_img = torch.cat((img, hmp), dim=0) 
        if self.is_training:
            stacked_img = self.data_transforms['train'](stacked_img) # Apply data augmentation
            img, hmp = stacked_img[:-1, :, :], stacked_img[-1:, :, :]
            # print(img.shape, hmp.shape) # torch.Size([3, 224, 224]) torch.Size([1, 224, 224])
        else:
            stacked_img = self.data_transforms['eval'](stacked_img) # Apply data augmentation
            img, hmp = stacked_img[:-1, :, :], stacked_img[-1:, :, :]
            
        img = (img - img.min()) / (img.max() - img.min())
        hmp = (hmp - hmp.min()) / (hmp.max() - hmp.min())
        
        if self.need_normalize:
            img = self.data_transforms['norm'](img)  # Apply ImageNet mean and std
        return img, hmp, label
                
    def __len__(self):
        return len(self.file_paths)
