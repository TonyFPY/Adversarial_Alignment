'''
Last modified by Jan. 21st, 2025
'''

import os
import numpy as np
import gc
import time
import argparse

import json
import glob
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import timm

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import model_loader, dataset, configs, utils
from attack import AdversarialAttack_Foolbox

def main(args):
    
    # Set GPU resources
    if args.cuda == -1: 
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.cuda}')
    print(device)
    
    # Define paths
    attack_name = args.attack
    data_path = '../../datasets/imagenet_big12/*.pth'
    results_path_avg = '../../results/' + 'adversarial_alignment_' + attack_name + '.csv'
    results_path_all = '../../results/adversarial_alignment/'
    os.makedirs(results_path_all, exist_ok=True)
    
    # Prepare for data
    imagenet_label_info_path = configs.IN_LABEL_INFO
    with open(imagenet_label_info_path, 'r') as file:
        imagenet_id2class = json.load(file)

    superclass2imagenet_mapper, imagenet2superclass_mapper, _ = utils.superclass_label_mapping(
        imagenet_id2class, configs.IMAGNET_BINARY_RANGES, need_other=False)
    
    BIG12_paths = glob.glob(os.path.join(configs.BIG12_DIR, '*.pth'))
    
    # Evaluation
    model_class = model_loader.TimmModels(file_path=args.file_path)
    num_models = model_class.get_num_models()
    for model_id, model_name, model_family in model_class.load_model_names(args.starting_from, args.ending_by):
        # Test model loader
        if model_family in ['test', 'test_dino_v1', 'test_dino_v2', 'test_adv', 'subset1', 'subset2']: continue
        # if model_family not in ['test']: continue
        
        utils.print_progress_info(f"Evaluating {model_id} {model_name} in {model_family} ...")
     
        start = time.time()
        
        model, _, _, data_config = model_class.load_model(
            model_family=model_family, model_name=model_name, pretrained=True, num_classes=1000, lc=True, device=device)
        model.eval()

        if hasattr(model, 'set_grad_checkpointing'): # efficientformer
            model.set_grad_checkpointing(False)
            print("Gradient checkpointing disabled.")

        # preprocessing module + model + classifier mapper
        wrapped_model = model_loader.CustomModelWrapper(model, data_config=data_config, mapper=superclass2imagenet_mapper)
        wrapped_model.eval()
        wrapped_model = wrapped_model.to(device)
        
        # Initialize adversarial attack
        adversarial_attack = AdversarialAttack_Foolbox(model=wrapped_model, attack_name=attack_name, device=device)            
        print(adversarial_attack.attack)
        
        # Specify post processing
        transform_post = transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC)
        
        # Create dataloader
        BIG12_val_dataset = dataset.ClickMe(
            BIG12_paths, model_name, imagenet2superclass_mapper=imagenet2superclass_mapper, need_normalize=False, is_training=False)
        BIG12_val_dataloader = DataLoader(BIG12_val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False, shuffle=False)
        
        norm_list, opt_epsilons, spearman_scores = [], [], []
        STOP = args.num_images
        num_valid_eval = 0
        for imgs, hmps, labels in tqdm(BIG12_val_dataloader):
            
            # Stop if the number of images exceeds the limit
            if num_valid_eval >= STOP: break
            
            # Apply attack
            adv_imgs, _, is_advs = adversarial_attack.run_attack(imgs, labels)
            
            # Evaluation
            with torch.no_grad():
                logits = wrapped_model(adv_imgs)
                adv_label_preds = torch.argmax(logits, dim=-1)
                
                logits = wrapped_model(imgs)
                label_preds = torch.argmax(logits, dim=-1)
                
            adv_imgs, imgs, labels, is_advs = adv_imgs.detach().cpu(), imgs.detach().cpu(), labels.detach().cpu(), is_advs.detach().cpu()
            # print(adv_imgs.shape, imgs.shape, labels.shape)

            # Postprocess the data to make a fair comparison
            if (imgs.shape[-1], imgs.shape[-2]) != (224, 224):
                imgs, adv_imgs = utils.resize_imgs(imgs, transform_post), utils.resize_imgs(adv_imgs, transform_post)
                
            # Compute the L2 norm
            norms = torch.norm(adv_imgs - imgs, p=2, dim=(1, 2, 3))

            # Compute spearman score
            atk_masks = torch.abs(torch.amax(adv_imgs - imgs, axis=1, keepdims=True)) # torch.Size([B, 1, 224, 224])
            # print(atk_masks.shape, atk_masks.max(), atk_masks.min())
            
            for i in range(args.batch_size):
                img, hmp, label, adv_img, label_pred, adv_label_pred, is_adv, atk_mask, norm = imgs[i], hmps[i], labels[i], adv_imgs[i], label_preds[i], adv_label_preds[i], is_advs[i], atk_masks[i], norms[i]
                
                # Failed prediction or failed attack
                if (label.item() != label_pred.item()) or (adv_label_pred.item() == label_pred.item()) or is_adv == False: continue
                num_valid_eval += 1
                
                # Compute spearman corr between ClickMe and attacking mask
                spearman_score = utils.spearman_correlation(hmp, atk_mask)
                
                # Store variables
                norm_list.append(norm)
                spearman_scores.append(spearman_score)
                
                # Save result of instance
                # print(type(label), type(adv_label_pred), type(norm), type(spearman_score))
                row_data = [
                    model_name, label.item(), adv_label_pred.item(), norm.item(), spearman_score.item()
                ]
                utils.write_csv_all(row_data, os.path.join(results_path_all, f"{model_name}_{attack_name}.csv"))
                del row_data
                                    
            # clear memory 
            del imgs, hmps, labels, adv_imgs, label_preds, adv_label_preds, is_advs, atk_masks, norms
            utils.clear_unused_memo("cuda:" + str(args.cuda), needClearGC=False)
        print("")
        
        end = time.time()
        
        # Save data
        norm_list, spearman_scores = torch.tensor(norm_list), torch.tensor(spearman_scores)
        eval_data = [
            model_name, num_valid_eval,
            round(torch.mean(norm_list).item(), 4), round(torch.std(norm_list).item(), 4), 
            round(torch.mean(spearman_scores).item(), 4), round(torch.std(spearman_scores).item(), 4), 
            int(end - start)
        ]
        utils.write_csv_avg(eval_data, results_path_avg)

        # clear memory
        del model, eval_data, norm_list, spearman_scores
        utils.clear_unused_memo("cuda:" + str(args.cuda), needClearGC=True)
        if utils.check_gpu_memory_usage_from_nvidia_smi(gpu_id = args.cuda, threshold=0.95):
            break
                    
    return

if __name__ == "__main__":
    # Create the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=int, default=0,
                        help="Enter a GPU device id from 0 to 7")
    parser.add_argument("-i", "--num_images", required=False, type=int,
                        default=1000, help="Number of images to be tested")
    parser.add_argument("-b", "--batch_size", required=False, type=int,
                        default=8, help="Number of images to be tested")
    parser.add_argument("-att", "--attack", required=False, choices=['L2FMNAttack', 'L2CarliniWagnerAttack', 'DDNAttack', 'L2BrendelBethgeAttack', 'L2DeepFoolAttack'], type=str,
                        default='L2FMNAttack', help="Please specify the attack")
    parser.add_argument("-st", "--starting_from", required=False, type=str, default='',
                        help="Please look up the /media/data_cifs/pfeng2/Harmoization/datasets/model_info/timm_models.csv and specify the model name to continue experiment")
    parser.add_argument("-eb", "--ending_by", required=False, type=str, default='',
                        help="Please look up the /media/data_cifs/pfeng2/Harmoization/datasets/model_info/timm_models.csv and specify the model name to continue experiment")
    parser.add_argument("-fp", "--file_path", required=False, choices=['full', 'subset'], type=str,
                        default='full', help="Please specify the file path for sub-list model testing")
    args = parser.parse_args()
    
    main(args)

    