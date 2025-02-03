'''
DINO model for ImageNet-1k  

Created on Oct. 4th, 2023
Last updated on Oct. 6th, 2023
Reference: https://github.com/facebookresearch/dino/blob/main/eval_linear.py
           https://github.com/facebookresearch/dinov2/tree/main#pretrained-heads---image-classification
Solutions to issue in dino v1:
    https://github.com/VITA-Group/SinNeRF/issues/6
'''

import os
import torch
import torch.nn as nn
from torchvision import models as torchvision_models

class DINOv2(nn.Module):
    def __init__(self, model_name, lc=True, num_labels=1000):
        super(DINOv2, self).__init__()
        # The keys are the model names from Timm library. 
        # The model name conversion is convenient for me to evaluate models in previous work.
        # It may be redundant for you and feel free to modify it.
        self.lc = lc 
        
        if self.lc:
            self.pairs = {
                'vit_small_patch14_dinov2.lvd142m': 'dinov2_vits14_lc',
                'vit_base_patch14_dinov2.lvd142m': 'dinov2_vitb14_lc',
                'vit_large_patch14_dinov2.lvd142m': 'dinov2_vitl14_lc',
                'vit_giant_patch14_dinov2.lvd142m': 'dinov2_vitg14_lc',
            }
        else:
            self.pairs = {
                'vit_small_patch14_dinov2.lvd142m': 'dinov2_vits14',
                'vit_base_patch14_dinov2.lvd142m': 'dinov2_vitb14',
                'vit_large_patch14_dinov2.lvd142m': 'dinov2_vitl14',
                'vit_giant_patch14_dinov2.lvd142m': 'dinov2_vitg14',
            }
        
        # print("Init DinoV2")

        # torch already has the pretrained models with classification layer
        self.model = torch.hub.load('facebookresearch/dinov2', self.pairs[model_name]) 

        # for _, p in self.model.named_parameters():
        #     print("RG:", p.requires_grad)# p.requires_grad = True

    def forward(self, x):
        output = self.model(x) # logits
        return output

class DINOv1(nn.Module):
    def __init__(self, model_name, lc=True, num_labels=1000):
        super(DINOv1, self).__init__()
        self.model_name = model_name
        self.lc = lc 
        self.pretrained_pairs =  {
            'vit_base_patch16_224.dino': ('facebookresearch/dino:main', 'dino_vitb16'),
            'vit_base_patch8_224.dino': ('facebookresearch/dino:main', 'dino_vitb8') ,
            'vit_small_patch16_224.dino':  ('facebookresearch/dino:main', 'dino_vits16'),
            'vit_small_patch8_224.dino': ('facebookresearch/dino:main', 'dino_vits8'),
            'resnet50.dino': "dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
            "xcit_small_12_p16.dino": "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth",
            # "xcit_small_12_p8.dino": "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth",
            "xcit_medium_24_p16.dino": "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
            # "xcit_medium_24_p8.dino": "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth",
        }

        self.linear_pairs =  {
            'vit_base_patch16_224.dino': {
                'url': 'dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth',
                'patch_size': 16,
                'n_last_blocks': 1,
                'avgpool_patchtokens': True
            },
            
            'vit_base_patch8_224.dino': {
                'url': 'dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth',
                'patch_size': 8,
                'n_last_blocks': 1,
                'avgpool_patchtokens': True
            },
            
            'vit_small_patch16_224.dino': {
                'url': 'dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth',
                'patch_size': 16,
                'n_last_blocks': 4,
                'avgpool_patchtokens': False
            },
            
            'vit_small_patch8_224.dino': {
                'url': 'dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth',
                'patch_size': 8,
                'n_last_blocks': 4,
                'avgpool_patchtokens': False
            },

            'resnet50.dino': {
                'url': "dino_resnet50_pretrain/dino_resnet50_linearweights.pth",
            },    

            "xcit_small_12_p16.dino": {
                'url': "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_linearweights.pth",
            },

            # "xcit_small_12_p8.dino":  {
            #     'url': "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_linearweights.pth",
            # },

            "xcit_medium_24_p16.dino": {
                'url': "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_linearweights.pth",
            },

            # "xcit_medium_24_p8.dino": {
            #     'url': "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_linearweights.pth",
            # },
        }

        self.model, self.embed_dim = self.loadDINOv1(model_name)
        self.linear_classifier = self.loadLC(model_name, self.embed_dim)

        '''
        final_model = nn.Sequential(self.lc, self.model.head.out)
        '''

    # load pretrained weights for DINO v1
    def loadDINOv1(self, model_name):

        assert model_name in list(self.pretrained_pairs.keys()), "Please check the model name"

        if 'vit' in model_name:
            x, y = self.pretrained_pairs[model_name]
            model = torch.hub.load(x, y)
            embed_dim = model.embed_dim
            return model, embed_dim
        # if the network is a XCiT
        elif "xcit" in model_name:
            model = torch.hub.load('facebookresearch/xcit:main', model_name.split('.dino')[0], num_classes=0)
            embed_dim = model.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif 'resnet50' in model_name:
            model = torchvision_models.__dict__['resnet50']()
            embed_dim = model.fc.weight.shape[1]
            model.fc = nn.Identity()
        else:
            print(f"Unknow architecture: {args.arch}")
            sys.exit(1)

        url = self.pretrained_pairs[model_name]
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
        return model, embed_dim

        # x, y = self.pretrained_pairs[model_name]
        # model = torch.hub.load(x, y)

        # # print(model)
        # if 'vit' in model_name:
        #     embed_dim = model.embed_dim
        # elif 'resnet50' in model_name:
        #     embed_dim = 2048
        #     model.fc = nn.Identity()
        # else:
        #     print('Please check the model_name or your model architecture is not supported for now.')
        #     return
        # return model, embed_dim

    # load pretrained linear weights for DINO v1
    def loadLC(self, model_name, embed_dim):
        if 'vit' in model_name:
            embed_dim = embed_dim * (self.linear_pairs[model_name]['n_last_blocks'] + int(self.linear_pairs[model_name]['avgpool_patchtokens']))

        cf = LinearClassifier(dim = embed_dim)
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + self.linear_pairs[model_name]['url'])["state_dict"]
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        cf.load_state_dict(state_dict, strict=True)

        return cf

    def forward(self, x):
        if "vit" in self.model_name:
            intermediate_output = self.model.get_intermediate_layers(x, self.linear_pairs[self.model_name]['n_last_blocks'])
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if self.linear_pairs[self.model_name]['avgpool_patchtokens']:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = self.model(x)
        
        if self.lc:
            output = self.linear_classifier(output) # logits
        return output

# classification head for Dino v1
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)