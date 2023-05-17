# Adversarial_Alignment

<p align="center">
<img src="./README/teaser.png"  style="width: 50%;"/>
</p>
<p><p><p>

Deep neural networks (DNNs) are known to have a fundamental sensitivity to adversarial attacks, perturbations of the input that are imperceptible to humans
yet powerful enough to change the visual decision of a model. Adversarial attacks have long been considered the “Achilles’ heel” of deep learning, which may eventually force a shift in modeling paradigms. Nevertheless, the formidable capabilities of modern large-scale DNNs have somewhat eclipsed these early concerns. Do adversarial attacks continue to pose a threat to DNNs?

## Dataset
Currently, we do our experiments on [ClickMe dataset](https://connectomics.clps.brown.edu/tf_records/), a large-scale effort for capturing feature importance maps from human participants that highlight parts that are relevant and irrelevant for recognition. The TF-Record file should be placed in `/datasets`. We create a subset of 'ClickMe', one image per category, in our experiment.

## Environment Setup

```
conda create -n adv python=3.8 -y
conda activate adv
pip install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
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

## Acknowledgement
This work relies heavily on [ClickMe](https://connectomics.clps.brown.edu/tf_records/) and [Harmonization](https://serre-lab.github.io/Harmonization/).

## License
The package is released under [MIT license](https://choosealicense.com/licenses/mit/)

