# Adversarial_Alignment

<p align="center">
<img src="./README/teaser.png"  style="width: 50%;"/>
</p>
<p><p><p>

## Abstract
Deep neural networks (DNNs) are known to have a fundamental sensitivity to adversarial attacks, perturbations of the input that are imperceptible to humans yet powerful enough to change the visual decision of a model. Adversarial attacks have long been considered the “Achilles' heel” of deep learning, which may eventually force a shift in modeling paradigms. Nevertheless, the formidable capabilities of modern large-scale DNNs have somewhat eclipsed these early concerns. Do adversarial attacks continue to pose a threat to DNNs?

In this study, we investigate how the robustness of DNNs to adversarial attacks has evolved as their accuracy on ImageNet has continued to improve. We measure adversarial robustness in two different ways: First, we measure the smallest adversarial attack needed to cause a model to change its object categorization decision. Second, we measure how aligned successful attacks are with the features that humans find diagnostic for object recognition. We find that adversarial attacks are inducing bigger and more easily detectable changes to image pixels as DNNs grow better on ImageNet, but these attacks are also becoming less aligned with the features that humans find diagnostic for object recognition. To better understand the source of this trade-off and if it is a byproduct of DNN architectures or the routines used to train them, we turned to the \emph{neural harmonizer}, a DNN training routine that aligns their perceptual strategies with humans. Harmonized DNNs achieve the best of both worlds and experience attacks that are both detectable and affect object features that humans find diagnostic for recognition, meaning that attacks on these models are more likely to be rendered ineffective by inducing similar effects on human perception. Our findings suggest that the sensitivity of DNNs to adversarial attacks could potentially be addressed by continued increases in model and data scale and novel training routines that promote alignment with biological intelligence.

## Dataset
We did our experiments on [ClickMe dataset](https://connectomics.clps.brown.edu/tf_records/), a large-scale effort for capturing feature importance maps from human participants that highlight parts that are relevant and irrelevant for recognition. We created a subset of ClickMe, one image per category, in our experiment. If you want to replicate our experiment, please put the TF-Record file in `./datasets`.

## Environment Setup

```
conda create -n adv python=3.8 -y
conda activate adv
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tensorflow==2.12.0
pip install timm==0.8.10.dev0
pip install harmonization
pip install numpy matplotlib scipy tqdm pandas
```

## Implementations
- You can enter the following command in Terminal
```
python main.py --model "resnet" --cuda 0 --spearman 1
```
- Google Colab notebook
    - You can run 2 .ipynb files if you have installation issues. Please check the folder `./scripts`


## Images 
- There are 10 example images in `./images`. 
- The images contains ImageNet images, human feature importance maps from ClickMe, and adversarial attacks for a variety of DNNs.

## Models
- In our experiment, 283 models have been tested
    - 125 PyTorch CNN models from [timm library](https://timm.fast.ai/)
    - 121 PyTorch ViT models from [timm library](https://timm.fast.ai/)
    - 15 PyTorch ViT/CNN hybrid architectures from [timm library](https://timm.fast.ai/)
    - 14 Tensorflow Harmonized models from [harmonizatin library](https://serre-lab.github.io/Harmonization/)
    - 4 Baseline models
    - 4 models that were trained for robustness to adversarial example
- The Top-1 ImageNet accuracy for each model refers to [Hugging Face results](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv)

## License
The package is released under [MIT license](https://choosealicense.com/licenses/mit/)

