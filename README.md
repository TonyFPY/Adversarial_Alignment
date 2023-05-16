# Adversarial_Alignment

## scripts
- Legacy 
    - 021 HFI Benchmark Pytorch.ipynb  
        - Reference code from Ivan
    - Adversarial_Attacks_ClickMe_FGSM.py 
        - Can be executed on lab node
        - Environment deployment is based on Model-VS-Human code repo
        - 79 PyTorch model
        - 4 Tensforflow model (To be implemented)
    - Results_FGSM.ipynb (Results Visualization)
- main.py
- model_collections.py
    - 245 models in total
- dataset.py
    - Load 1000 ClickMe data
- parameters.py
    - Parameters for attack strategy
- util.py
    - Spearman correlation
    - L-2 norm
    - L-inf norm
    - Write results into .csv files
- ClickMe_test_1000.py  
    - Extract 1000 images from the original ClickMe test data
    - 1 image for 1 class, 1000 images in total
- Adversarial_Attacks_Test.ipynb
    - A test file to test different attacks
- Attack-VS-Human.ipynb
    - A test file to test attacks regions vs. human attention maps
- Results_Visualization.ipynb
    - L-2 norm vs. ImageNet Top-1 Accuracy
    - L-inf norm vs. ImageNet Top-1 Accuracy
    - Spearman correlation vs. ImageNet Top-1 Accuracy

## results 
- FGSM.csv (a temp result)

## datasets
- imagenet_acc.csv
    - Extracted from [pytorch-image-models](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv)
- clickme_test.tfrecords (Orginial ClickMe testing dataset, 17,000+ images in total)
    - please download via [link](https://drive.google.com/file/d/1-0qjq7LYGokmpXs9e6G9DbUvZxJJUAdm/view?usp=share_link)
- clickme_test_1000.tfrecords (We use this one, 1 image for 1 class, 1000 images in total)
    - please execute **ClickMe_test_1000.py**

## papers
- Is Robustness the Cost of Accuracy? ([GitHub](https://github.com/huanzhang12/Adversarial_Survey), [Paper](https://arxiv.org/abs/1808.01688))
