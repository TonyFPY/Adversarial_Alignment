'''
python Adversarial_Attacks_Distortion.py --model "resnet" --cuda "0" --lpips "vgg"
python Adversarial_Attacks_Distortion.py --model "vgg" --cuda "1" --lpips "vgg"
python Adversarial_Attacks_Distortion.py --model "efficientnet" --cuda "2" --lpips "vgg"
python Adversarial_Attacks_Distortion.py --model "vit_b16" --cuda "3" --lpips "vgg"
python Adversarial_Attacks_Distortion.py --model "convnext" --cuda "4" --lpips "vgg"
python Adversarial_Attacks_Distortion.py --model "maxvit" --cuda "5" --lpips "vgg"
'''

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import timm
import model_collections
import torchattacks
import argparse
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips

import util
import model_collections
import parameters
import dataset

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def attack(model, img, target, eps, alpha):
    # Initialize an attack
    attack_obj = torchattacks.FFGSM(model=model, eps=eps, alpha=alpha) 
    # attack_obj = torchattacks.BIM(model, eps=eps, alpha=10/255, steps=20)
    perturbed_img = attack_obj(img, target)

    # Re-classify the perturbed image
    with torch.no_grad():
        output = model(perturbed_img)
        pred = torch.argmax(output, axis=-1) # get the index of the max log-probability

    return perturbed_img, pred

def main():
    # Create the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda",
                        required=True,
                        choices=["0","1","2","3","4","5","6","7"],
                        help="Enter a GPU device id from 0 to 7")
    parser.add_argument("-m", "--model",
                        required=True,
                        choices=[
                            "vit", "vgg", "resnet", "efficientnet", "convnext", "mobilenet",
                            "inception", "densenet", "regnet", "xception", "mobilevit",
                            "swin", "mixnet", "dpn", "darknet", "maxvit", "beit", "vit_clip",
                            "volo", "deit", "nfnet", "cait", "xcit", "tinynet", "lcnet", "dla", 
                            "mnasnet", "coatnet", "csp",
                            "resize_group_0", "resize_group_1", "resize_group_2", "resize_group_3", "resize_group_4"
                        ],
                        help="Please see util.py")
    parser.add_argument("-l", "--lpips",
                        required=True,
                        choices=['None', 'alex', 'vgg', 'squeeze'],
                        help="Please see util.py")
    parser.add_argument("-r", "--resize",
                        required=True,
                        choices=[1, 0],
                        type = int,
                        help="Some models input needs resized")
    parser.add_argument("-s", "--spearman",
                        required=True,
                        choices=[1, 0],
                        type = int,
                        help="Compute spearman correlation between adversarial masks and heat maps")
    args = parser.parse_args()

    # Set GPU resources
    device = torch.device("cuda:" + args.cuda)

    # Get models to be experimented
    mCollections = model_collections.Models()
    model_type = args.model
    model_names = mCollections.get_model_families(model_type)
    print(mCollections.get_num_models(model_type))

    # Get parameters for FFGSM
    para = parameters.Parameters()
    epsilons = para.get_epsilons()
    alpha = para.get_alpha()

    # Initialize lpips
    if args.lpips != "None":
        PerceptualSimilarity = lpips(net_type=args.lpips).to(device)

    # Define paths
    case = "FFGSM_spearman_corr"
    data_path = '../datasets/clickme_test_1000.tfrecords'
    results_path_avg = '../results/' + case + '.csv'
    # results_path_avg = '../results/BIM_distortion.csv'
    # results_path_all = '../results/FFGSM_distortion/FFGSM_distortion_' + model_type + '.csv'

    offset = 0
    for i, model_name in enumerate(model_names[offset:]):
        print(str(i + offset), model_name)
        
        folder_path = '../results/' + case 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
        results_path_all = folder_path + '/' + case + '_' + model_name + '.csv'
        # results_path_all = '../results/BIM_distortion/BIM_distortion_' + model_name + '.csv'

        # Model Instantiation
        try:
            model = timm.create_model(model_name, num_classes=1000, pretrained=True).to(device)
        except:
            continue

        model.eval()
        
        # Instantiate a tensorflow dataset
        data = dataset.Dataset(data_path)
        l2_list, linf_list = [], []
        opt_epsilons = []
        lpips_scores = []
        spearman_scores = []
        total_cnt, init_correct, aa_correct = 0, 0, 0

        for imgs, hmps, labels in data.get_dataset(10, False): # image, heatmap, label
            imgs = util.tf2torch(imgs).to(device)
            hmps = util.tf2torch(hmps)

            if args.resize == 1:
                items = model_name.split('_') # usually the last item is a number
                size = items[-1]
                for i in range(len(items)-1, -1, -1):
                    size = items[i]
                    if size.isdigit():
                        break
                size = int(size)
                # print(size)
                transform = transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC)
                imgs = transform(imgs)
                hmps = transform(hmps)

            imgs = util.img_normalize(imgs)
            hmps = util.img_normalize(hmps)
            labels = util.tf2torch(labels).to(device)

            for img, hmp, label in zip(imgs, hmps, labels):
                # print(img.shape, hmp.shape, logit.shape)

                # Add a dimension (batch == 1)
                img = torch.unsqueeze(img, 0) # (1, 3, H, W)
                hmp = torch.unsqueeze(hmp, 0) # (1, 1, H, W)
                # print(hmp.shape)
                label = torch.unsqueeze(label, 0)
                target = torch.argmax(label, axis=-1) # tensor([343], device='cuda:0')

                # print(img.shape, target.shape)

                # Forward pass the data through the model
                with torch.no_grad():
                    output = model(img)
                    init_pred = torch.argmax(output, axis=-1) # get the index of the max log-probability

                # If the initial prediction is wrong, just move on
                total_cnt += 1
                print("\rSearching optimal epsilon for image: %s | %s" % (
                    str(total_cnt), 
                    str(1000)), end=" ")
                if init_pred.item() != target.item():
                    continue
                init_correct += 1
                
                # key: eps; val: perturbed_img
                info = {} # Only store one key-val pair
                key = None

                # Apply first attack
                initial_eps = 0.001
                initial_eps_id = epsilons.index(initial_eps)
                perturbed_img, perturbed_pred = attack(model=model, img=img, target=target, eps=initial_eps, alpha=alpha)

                # Define search boundary
                if perturbed_pred.item() == target.item():
                    l, r = initial_eps_id+1, len(epsilons)-1
                else:
                    l, r = 0, initial_eps_id
                    key = initial_eps
                    info[initial_eps] = perturbed_img

                # Apply binary search, find the first epsilon that causes the successful attack
                while l < r:
                    m = l + (r - l) // 2
                    eps = epsilons[m]
                    perturbed_img, perturbed_pred = attack(model=model, img=img, target=target, eps=eps, alpha=alpha)
                    if perturbed_pred.item() == target.item():
                        l = m + 1
                    else:
                        r = m
                        if not key or key > eps:
                            if key:
                                info.pop(key)
                            info[eps] = perturbed_img
                            key = eps

                # Get optimal eps
                optimal_eps = epsilons[l]
                opt_epsilons.append(optimal_eps)
                if not optimal_eps in info:
                    perturbed_img, _ = attack(model=model, img=img, target=target, eps=optimal_eps, alpha=alpha)
                else:
                    perturbed_img = info[optimal_eps]

                # Store l2, linf
                a, b = perturbed_img.cpu().numpy(), img.cpu().numpy()
                l2, linf = util.l2_loss(a, b), util.linf_loss(a, b)
                l2_list.append(l2)
                linf_list.append(linf)

                # LPIPS 
                if args.lpips != "None":
                    lpips_score = PerceptualSimilarity(perturbed_img, img).item() if args.lpips != None else -1
                else:
                    lpips_score = 0
                lpips_scores.append(lpips_score)

                # Spearman correlation
                if args.spearman == 1:
                    mask = np.abs(np.mean(a - b, axis=1, keepdims=True)) # (1, 1, 224, 224)
                    spearman_score = util.spearman_correlation(mask, hmp.numpy())
                    spearman_scores.append(spearman_score)

                # Save the data into 
                row_data = [
                    model_name, str(total_cnt-1), str(round(optimal_eps, 5)), 
                    str(l2), str(linf), str(lpips_score), str(spearman_score)
                ]
                util.write_CSV_all(row_data, results_path_all)

        print("")
        
        # Save data
        row_data = [
            model_name, str(init_correct), str(np.mean(opt_epsilons)), 
            str(np.mean(l2_list)), str(np.mean(linf_list)), 
            str(np.mean(lpips_scores)), str(np.mean(spearman_scores))
        ]
        util.write_CSV_avg(row_data, results_path_avg)

        torch.cuda.empty_cache()
   
if __name__ == '__main__':
    main()