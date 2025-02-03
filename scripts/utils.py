
import gc
import os

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import subprocess

import torch
import torch.nn as nn
import torch.nn. functional as F
from torch.utils.data import DataLoader
from torchmetrics import SpearmanCorrCoef
from statistics import mode, mean
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

import configs

def superclass_label_mapping(ori_class_info, superclass_info, need_other=True):

    '''
    superclass2imagenet_mapper = {
        0: ('dog', array([151, 152, 153, 154, ...])),
        ...
    } 
    '''
    superclass2imagenet_mapper = {}
    imagenet_ids = set(range(0, 1000))
    imagenet_superclass_id_set = set()
    size = len(superclass_info.keys())

    for idx, (superclass_name, id_sets) in enumerate(superclass_info.items()): # 0, ('dog', [151, ..., 268])

        superclass2imagenet_mapper[idx] = (superclass_name, np.array(list(id_sets)))
        imagenet_superclass_id_set.update(id_sets)

    rest = imagenet_ids.difference(imagenet_superclass_id_set)
    superclass2imagenet_mapper[size] = ('other', np.array(list(rest)))

    '''
    imagenet2superclass_mapper = {
        151: ("n02085620", "Chihuahua", 0, "dog"),
        ...
    }
    '''
    imagenet2superclass_mapper = {}
    for superclass_index, (superclass_name, imagenet_indices) in superclass2imagenet_mapper.items():
        for imagenet_index in imagenet_indices:

            imagenet_index_str = str(imagenet_index)
            if imagenet_index_str in ori_class_info: # Retrieve ImageNet ID and class name using the string index
                imagenet_id, class_name = ori_class_info[imagenet_index_str]
                
                # Map the integer index to the desired tuple
                imagenet2superclass_mapper[imagenet_index] = (imagenet_id, class_name, superclass_index, superclass_name)

    if not need_other: superclass2imagenet_mapper.pop(size)
    return superclass2imagenet_mapper, imagenet2superclass_mapper, imagenet_superclass_id_set

def evaluate_model(data_loader, model, criterion, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():  # Disable gradient computation
        for images, _, labels in tqdm(data_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            
            # Calculate the batch loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)  # Multiply by batch size

            # Convert output probabilities to predicted class
            _, preds = torch.max(outputs, 1)
            
            # Compare predictions to true label
            total_correct += (preds == labels).sum().item()
            total_images += labels.size(0)

    # Calculate average losses and accuracy
    avg_loss = total_loss / total_images
    accuracy = total_correct / total_images * 100

    print(f'Evaluation Results - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def show_images(img, ax=None, p=False, smooth=False, **kwargs):
    # Process the image
    try:
        img = img.detach().cpu()
    except:
        img = np.array(img)

    img = np.array(img, dtype=np.float32)

    if len(img.shape) == 4: #[1, 3, 224, 224]
        img = img[0]  #[3, 224, 224]

    # print(img.shape)

    # Check channel order and adjust if necessary
    if img.shape[0] == 1:             #[1, 224, 224] heatmap
        img = img[0]                  #[224, 224]  
    elif img.shape[0] == 3:           #[3, 224, 224] image
        img = np.moveaxis(img, 0, -1) #[224, 224, 3]

    # Normalize the image
    if img.max() > 1 or img.min() < 0:
        img -= img.min()
        img /= img.max()

    # Apply percentile clipping if specified
    if p is not False:
        img = np.clip(img, np.percentile(img, p), np.percentile(img, 100-p))

    # Apply Gaussian smoothing if specified
    if smooth and len(img.shape) == 2:
        img = gaussian_filter(img, smooth)
        
    # Display the image on the specified axes
    ax.imshow(img, **kwargs)
    ax.axis('off')

def get_alpha_cmap(cmap):
    if isinstance(cmap, str):
      cmap = plt.get_cmap(cmap)
    
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    alpha_cmap = ListedColormap(alpha_cmap)

    return alpha_cmap

def compute_saliency_map(model, images, labels):
    device = model.device
    images, labels = images.to(device), labels.to(device)
    
    model.eval()
    
    # compute prediction and loss
    images.requires_grad = True
    output = model(images)
    
    # get correct class scores
    correct_class_scores = output.gather(1, labels.view(-1, 1)).squeeze()
    ones_tensor = torch.ones(correct_class_scores.shape).to(device) # scores is a tensor here, need to supply initial gradients of same tensor shape as scores.
    correct_class_scores.backward(ones_tensor, retain_graph=True) # compute the gradients while retain the graph
    
    # compute saliency maps
    grads = torch.abs(images.grad)
    saliency_maps, _ = torch.max(grads, dim=1, keepdim=True) # saliency map (N, C, H, W) -> (N, 1, H, W)
    
    images.grad.zero_() # reset the gradients
    
    return saliency_maps

def spearman_correlation(heatmaps_a, heatmaps_b):
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must" \
                                                 "have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (1, W, H)."

    HUMAN_SPEARMAN_CEILING = 0.65753
    spearman = SpearmanCorrCoef(num_outputs=1)
    heatmaps_a, heatmaps_b = heatmaps_a.reshape(1, -1).squeeze(), heatmaps_b.reshape(1, -1).squeeze()
    score = spearman(heatmaps_a, heatmaps_b) / HUMAN_SPEARMAN_CEILING
    
    return score

class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Args:
            model: a base model to get CAM which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data

        cam = self.getCAM(self.values, weight_fc, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data

class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data

class GradCAMpp(CAM):
    """ Grad CAM plus plus """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAMpp(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[0, idx].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data

class SmoothGradCAMpp(CAM):
    """ Smooth Grad CAM plus plus """

    def __init__(self, model, target_layer, n_samples=25, stdev_spread=0.15):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
            n_sample: the number of samples
            stdev_spread: standard deviationß
        """

        self.n_samples = n_samples
        self.stdev_spread = stdev_spread

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        stdev = self.stdev_spread / (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev

        indices = []
        probs = []

        for i in range(self.n_samples):
            self.model.zero_grad()

            x_with_noise = torch.normal(mean=x, std=std_tensor)
            x_with_noise.requires_grad_()

            score = self.model(x_with_noise)

            prob = F.softmax(score, dim=1)

            if idx is None:
                prob, idx = torch.max(prob, dim=1)
                idx = idx.item()
                probs.append(prob.item())

            indices.append(idx)

            score[0, idx].backward(retain_graph=True)

            activations = self.values.activations
            gradients = self.values.gradients
            n, c, _, _ = gradients.shape

            # calculate alpha
            numerator = gradients.pow(2)
            denominator = 2 * gradients.pow(2)
            ag = activations * gradients.pow(3)
            denominator += \
                ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator))
            alpha = numerator / (denominator + 1e-7)

            relu_grad = F.relu(score[0, idx].exp() * gradients)
            weights = \
                (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

            # shape => (1, 1, H', W')
            cam = (weights * activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)

            if i == 0:
                total_cams = cam.clone()
            else:
                total_cams += cam

        total_cams /= self.n_samples
        idx = mode(indices)
        prob = mean(probs)

        print("predicted class ids {}\t probability {}".format(idx, prob))

        return total_cams.data, idx

    def __call__(self, x):
        return self.forward(x)

def get_xai_map(model, image, label, target_layer, method='saliency_map'):
    if method == 'saliency_map':
        return compute_saliency_map(model, image, label)
    else:
        if method == 'cam':
            wrapped_model = CAM(model, target_layer)
        elif method == 'grad_cam':
            wrapped_model =GradCAM(model, target_layer)
        elif method == 'grad_cam_++':
            wrapped_model = GradCAMpp(model, target_layer)
        elif method == 'smooth_grad_cam_++':
            wrapped_model = SmoothGradCAMpp(model, target_layer, n_samples=25, stdev_spread=0.15)
            
        heatmap, idx = wrapped_model(image)
        _, _, H, W = image.shape
        heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)
        heatmap = 255 * heatmap.squeeze()
        return heatmap
    return None
    
def resize_imgs(imgs, transform_post):
    """
    Applies the transform_post transformation to a batch of images imgs.

    Args:
        imgs (torch.Tensor): Batch of images of shape (B, C, H, W).
        transform_post (torchvision.transforms.Resize): The Resize transformation.

    Returns:
        torch.Tensor: Batch of transformed images of shape (B, C, H_new, W_new).
    """
    # Check if imgs is a 4D tensor
    if imgs.dim() != 4:
        raise ValueError("imgs should be a 4D tensor of shape (B, C, H, W)")

    # Extract the target size from the transform_post
    size = transform_post.size  # Can be an int or tuple
    if isinstance(size, int):
        size = (size, size)

    # Map the interpolation mode from transforms to F.interpolate mode
    interpolation_mode = {
        transforms.InterpolationMode.NEAREST: 'nearest',
        transforms.InterpolationMode.BILINEAR: 'bilinear',
        transforms.InterpolationMode.BICUBIC: 'bicubic',
        transforms.InterpolationMode.TRILINEAR: 'trilinear',
        transforms.InterpolationMode.AREA: 'area',
        transforms.InterpolationMode.NEAREST_EXACT: 'nearest-exact',
    }
    mode = interpolation_mode.get(transform_post.interpolation, 'bilinear')

    # Ensure imgs is a float tensor
    imgs = imgs.float()

    # Apply the resize transformation using F.interpolate
    imgs_resized = F.interpolate(imgs, size=size, mode=mode, align_corners=False)

    return imgs_resized

def clear_unused_memo(device_name, needClearGC):
    if needClearGC:
        gc.collect()
    
    if device_name:
        with torch.cuda.device(device_name):
            torch.cuda.empty_cache()
            
def check_cuda_memo_usage(device, threshold=0.9):
    total_mem = torch.cuda.get_device_properties(device).total_memory
    current_mem = torch.cuda.memory_allocated(device)
    usage_ratio = current_mem / total_mem
    return True if usage_ratio > threshold else False

def check_gpu_memory_usage_from_nvidia_smi(gpu_id=0, threshold=0.9):
    """Get GPU memory usage for a specific GPU.
    Parameters:
        - gpu_id (int): The GPU ID to get memory usage for.
    Returns:
        - used_memory (int): Amount of memory used in MiB.
        - total_memory (int): Total available memory in MiB.
    """
    
    # Querying nvidia-smi to get the memory usage
    query = f'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader --id={gpu_id}'
    result = subprocess.check_output(query, shell=True).decode('utf-8')
    
    # Parsing the result
    used_memory, total_memory = map(int, result.strip().split(','))
    
    # Compute usage ratio
    usage_ratio = used_memory / total_memory
    
    return True if usage_ratio > threshold else False

def write_csv_all(record, path):
    header = ['model', 'label', 'adv_pred', 'norm', 'spearman']
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)

def write_csv_avg(record, path):
    header = [
        'model', 'num_valid_eval', 
        'avg_norm', 'std_norm',
        'avg_spearman', 'std_spearman', 'time',
    ]
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)
        
def find_model_info(path, model_name):
    df = pd.read_csv(path)
    index = df[df['model'] == model_name].index.tolist()
    return df.loc[index[0]].tolist()

''' Set print log '''
def print_progress_info(txt, total_length=80):
    # Length of the text to be inserted
    txt_length = len(txt)

    # Calculate the number of asterisks needed
    num_asterisks = total_length - txt_length - len("******************************************")

    # Construct the line
    line = f"{'*' * (num_asterisks // 2 + 8)} {txt} {'*' * (num_asterisks // 2 + 8)}"

    # Print the line
    print(line)

if __name__ == "__main__":
    pass