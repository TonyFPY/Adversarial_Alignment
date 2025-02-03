import os
import glob
import time
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_data_config, resolve_model_data_config
from timm.data.transforms_factory import create_transform

from dino import DINOv1, DINOv2

''' TIMM models '''
class TimmModels:
    def __init__(self, file_path="full"):
        self.model_families = [
            'test', 'test_adv', 'test_dino_v1', 'test_dino_v2', 'subset1', 'subset2',
            'beit', 'caformer', 'cait', 'coat', 'coatnet', 'convit', 'convmixer', 'davit', 'deit', 'deit3',
            'densenet', 'dla', 'dpn', 'efficientformer', 'efficientnet', 'focalnet', 'hrnet', 'lcnet', 'levit', 'maxvit', 
            'mixnet', 'mobilenet', 'mvit', 'pit', 'poolformer', 'pvt', 'regnet', 'repvgg', 'res2net', 'resnest', 
            'resnet', 'rexnet', 'sknet', 'swin', 'tresnet', 'twins', 'vgg', 'visformer', 'vit_base', 'vit', 'vit_large',
            'vit_relpos', 'volo', 'xcit', 'convnext', 'convnext_v2', 'crossvit', 'tf_mobilenet', 'mobilevit', 'seresnet',
            'mnasnet', 'swin_v2', 'eva', 'eva_v2', 'tinynet', 'other', 'resnet18_l2_adv', 'resnet50_l2_adv', 'wide_resnet50_2_l2_adv',
            'wide_resnet50_4_l2_adv', 'resnet18_linf_adv', 'resnet50_linf_adv', 'wide_resnet50_2_linf_adv', 'other_adv',
            'clip_1', 'clip_2', 'dino_v1', 'dino_v2'
        ]
        
        fp = "/media/data_cifs/pfeng2/Harmoization/datasets/model_info/timm_models.csv" if file_path=="full" else  "/media/data_cifs/pfeng2/Harmoization/datasets/model_info/timm_models_subset.csv"
        print(fp)
        self.df = pd.read_csv(fp)
        self.df.insert(0, 'id', range(0, len(self.df)))
        

    def get_num_models(self):
        return len(self.df)
    
    def check_timm_models_exists(self, name):
        model = timm.list_models(name, pretrained=True)
        return len(model)
    
    def get_model_families(self):
        return self.model_families 
            
    def get_family_model_names(self, name):
        if self.df['superclass'].isin([name]).any():
            return self.df[self.df["superclass"]==name]["model_name"].tolist()
        return []
    
    def get_all_model_names(self):
        return self.df["model_name"].tolist()
    
    def get_model_names_starting_from_family(self, model_family):
        idx = self.model_families.index(model_family)
        cur_model_families = self.model_families[idx:]
        model_names = []
        for cur_mf in cur_model_families:
            # print(cur_mf)
            model_names += self.get_family_model_names(cur_mf)
        
        return model_names
    
    def load_model_names(self, starting_from='', ending_by=''):
        start = starting_from == ''  # Indicates whether we should start yielding
        end = False  # Indicates whether we should stop yielding

        # Check if the starting model name is in the DataFrame if specified
        if not start and starting_from not in self.df['model_name'].values:
            raise ValueError(f"Model name '{starting_from}' not found in the list of models.")

        # Check if the end_by model name is in the DataFrame if specified
        if ending_by and ending_by not in self.df['model_name'].values:
            raise ValueError(f"Model name '{ending_by}' not found in the list of models.")

        # Ensure starting_from comes before ending_by if both are specified
        testing_classes = ['test', 'test_dino_v1', 'test_dino_v2', 'test_adv', 'subset1', 'subset2']
        if starting_from and ending_by:
            start_idx = self.df.index[(self.df['model_name'] == starting_from) & (~self.df['superclass'].isin(testing_classes))].tolist()[0]
            end_idx = self.df.index[(self.df['model_name'] == ending_by) & (~self.df['superclass'].isin(testing_classes))].tolist()[0]      
            
            if start_idx > end_idx:
                raise ValueError(f"Model name '{starting_from}' must appear before '{ending_by}'.")

        for model_id, model_name, model_family in zip(self.df['id'], self.df['model_name'], self.df['superclass']):
            if model_family in testing_classes: continue
            
            if not start:
                if model_name == starting_from:
                    start = True
                else:
                    continue

            yield model_id, model_name, model_family
            
            # If we reach the ending_by model, we stop yielding
            if ending_by and model_name == ending_by:
                break
    
    def load_adv_model(self, model_name, lc=True, models_dir='/media/data_cifs/pfeng2/Adversarial_Alignment/models/'):
        ckpt_path = os.path.join(models_dir, model_name + '.ckpt')
        assert os.path.exists(ckpt_path), 'The model does not exist.'
        print('[', ckpt_path, '] is found!')
        
        if model_name.startswith('resnet18'):
            from torchvision.models import resnet18
            model = resnet18(pretrained=False)
            
        if model_name.startswith('resnet50'):
            from torchvision.models import resnet50
            model = resnet50(pretrained=False)
            
        if model_name.startswith('wide_resnet50_2'):
            from torchvision.models import wide_resnet50_2
            model = wide_resnet50_2(pretrained=False)
            
        checkpoint = torch.load(ckpt_path)
        sd = {k[len('module.model.'):]:v for k,v in checkpoint['model'].items() if k[:len('module.model.')] == 'module.model.'}  # Consider only the model and not normalizers or attacker
        model.load_state_dict(sd)
        if not lc:
            model.fc = torch.nn.Identity()
        return model

    def load_dinov1(self, model_name, lc=True):
        return DINOv1(model_name, lc=lc)

    def load_dinov2(self, model_name, lc=True):
        return DINOv2(model_name, lc=lc)
    
    def load_model(self, model_family='resnet', model_name='resent18.tv_in1k', pretrained=True, num_classes=1000, lc=True, device=torch.device('cpu')):
        # print(num_classes == 0 and lc == True, num_classes > 0 and lc == False)
        assert not (num_classes == 0 and lc == True) and not (num_classes > 0 and lc == False), "Issues with classifier"
        
        # Load models
        if model_family.endswith('adv'):
            model = self.load_adv_model(model_name=model_name, lc=lc).to(device)
        elif 'dino_v1' in model_family:
            model = self.load_dinov1(model_name=model_name, lc=lc).to(device)
        elif 'dino_v2' in model_family:
            model = self.load_dinov2(model_name=model_name, lc=lc).to(device)
        elif 'hmn' in model_family:
            mn = model_name.split('_harmonized')[0]
            model = timm.create_model(mn, num_classes=1000, pretrained=False)
            root_path = '/media/data_cifs/projects/prj_pseudo_clickme/Checkpoints/for_adv'
            ckpt_path = os.path.join(root_path, f'{model_name}.pth.tar')
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
        else:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes).to(device)
        
        # Extract model configurations
        data_config = resolve_model_data_config(model)
        img_transform_training = create_transform(**data_config, is_training=True)
        img_transform_eval = create_transform(**data_config, is_training=False)

        return model, img_transform_training, img_transform_eval, data_config

def load_timm_model(model_name='resent18.tv_in1k', ckpt_path=None, pretrained=True, num_classes=1000, device=torch.device('cpu')):
    if not ckpt_path:
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)
    else:
        if os.path.isfile(ckpt_path):
            model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
            model = timm.create_model(model_name, num_classes=num_classes, pretrained=True)
            
    model = model.to(device)

    data_config = resolve_model_data_config(model)
    img_transform_training = create_transform(**data_config, is_training=True)
    img_transform_eval = create_transform(**data_config, is_training=False)

    return model, img_transform_training, img_transform_eval, data_config

def load_normalize_model(model, data_config):
    """
    Adds preprocessing steps to the model's forward method using data_config.

    Args:
        model (nn.Module): The model to which preprocessing will be added.
        data_config (dict): The data configuration dictionary containing parameters like 'input_size', 'mean', 'std', etc.

    Returns:
        nn.Module: The model with preprocessing included in the forward pass.
    """

    # Store the original forward method of the model
    original_forward = model.forward

    # Extract parameters from data_config
    input_size = data_config.get('input_size', (3, 224, 224))
    # crop_pct = data_config.get('crop_pct', 0.875)
    interpolation = data_config.get('interpolation', 'bilinear')
    mean = data_config.get('mean', (0.485, 0.456, 0.406))
    std = data_config.get('std', (0.229, 0.224, 0.225))

    # # Compute resize size based on crop percentage
    # crop_size = input_size[1:]  # (H, W)
    # resize_size = int(math.floor(crop_size[0] / crop_pct))

    # Prepare mean and std tensors
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    # Interpolation mode mapping
    interpolation_modes = {
        'bilinear': 'bilinear',
        'nearest': 'nearest',
        'bicubic': 'bicubic',
    }
    mode = interpolation_modes.get(interpolation, 'bilinear')
    
    device = next(model.parameters()).device

    # Define a function that will replace the model's forward method
    def forward_with_preprocessing(x):

        x = x.to(device)

        # Resize
        x = F.interpolate(x, size=resize_size, mode=mode, align_corners=False)

        # # Center crop
        # x = center_crop(x, output_size=crop_size)

        # Normalize
        x = (x - mean.to(device)) / std.to(device)

        # Call the original forward method
        return original_forward(x)

    # Replace the model's forward method with the new one
    model.forward = forward_with_preprocessing

    return model

    
def load_superclass_mapped_model(model, mapper, style='max'):
    """
    Modifies a model's forward method to map its output classes into superclasses using specified aggregation style.

    Args:
        model (torch.nn.Module): The neural network model whose forward method will be modified.
        mapper (dict): A dictionary mapping superclasses to sets of indices corresponding to the original model's outputs.
        style (str): Determines the aggregation style. If 'max', the maximum output among mapped classes is taken.
                     If any other string is provided, the sum of the outputs is used.

    Returns:
        torch.nn.Module: The modified model with a new forward method that aggregates outputs into superclasses.
    """
    
    # Store the original forward method of the model
    original_forward = model.forward
    
    def forward_wrapper(forward):
        
        def forwarded(*args, **kwargs):
            
            x = forward(*args, **kwargs) # Execute the original forward method
            
            # Check if 'with_latent' keyword argument is provided and true
            if 'with_latent' in kwargs and kwargs['with_latent']:
                return x
            
            # Aggregate the outputs based on the provided style
            if style == 'max': # this is better
                # Use max pooling across the indices specified for each superclass in the mapper
                x = torch.stack([x[:, imagenet_ids].max(1).values for _, imagenet_ids in mapper.values()], dim=1)
            else:
                # Use sum pooling across the indices specified for each superclass in the mapper
                x = torch.stack([x[:, imagenet_ids].sum(dim=1) for _, imagenet_ids in mapper.values()], dim=1)
                
            return x
        
        return forwarded

    model.forward = forward_wrapper(original_forward)
    
    return model

class CustomModelWrapper(nn.Module):
    def __init__(self, model, data_config=None, mapper=None, style='max'):
        super(CustomModelWrapper, self).__init__()
        self.model = model
        self.data_config = data_config  # Data configuration dictionary
        self.mapper = mapper
        self.style = style
        self.device = next(model.parameters()).device

        # Extract preprocessing parameters from data_config
        if data_config is not None:
            # Get the expected input size (assuming format (C, H, W))
            self.input_size = data_config.get('input_size', (3, 224, 224))[1:]  # (H, W)
            # Get mean and std for normalization
            mean = data_config.get('mean', (0.485, 0.456, 0.406))
            std = data_config.get('std', (0.229, 0.224, 0.225))
            self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)
            # Get interpolation mode
            interpolation = data_config.get('interpolation', 'bilinear')
            self.interpolation_modes = {
                'bilinear': 'bilinear',
                'nearest': 'nearest',
                'bicubic': 'bicubic',
                # Add other modes if needed
            }
            self.mode = self.interpolation_modes.get(interpolation, 'bilinear')
        else:
            # Default preprocessing parameters
            self.input_size = (224, 224)
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            self.mode = 'bilinear'

    def forward(self, x):
        # Ensure input requires gradient for saliency maps
        x = x.to(self.device)

        # Resize directly to model's expected input size without center cropping
        x = F.interpolate(x, size=self.input_size, mode=self.mode, align_corners=False)

        # Normalize
        x = (x - self.mean) / self.std

        # Process the input through the original model
        y = self.model(x)

        # Apply superclass mapping if it is defined
        if self.mapper:
            if 'max' in self.style:
                y = torch.stack(
                    [y[:, imagenet_ids].max(dim=1).values for _, imagenet_ids in self.mapper.values()],
                    dim=1
                )
            else:
                y = torch.stack(
                    [y[:, imagenet_ids].sum(dim=1) for _, imagenet_ids in self.mapper.values()],
                    dim=1
                )

        return y
    
if __name__ == "__main__":
    model_class = TimmModels()
    starting_from = 'vit_large_patch16_224.augreg_in21k_ft_in1k'
    ending_by = 'vit_relpos_small_patch16_224.sw_in1k'
    for model_id, model_name, model_family in model_class.load_model_names(starting_from=starting_from, ending_by=ending_by):
        print(model_id, model_name, model_family)
