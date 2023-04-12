# Adverserial_Attacks_ClickMe

## scripts
- 021 HFI Benchmark Pytorch.ipynb  
    - Reference code from Ivan
- Adversarial_Attacks_ClickMe_FGSM.py 
    - Can be executed on lab node
    - Environment deployment is based on Model-VS-Human code repo
    - 79 PyTorch model
    - 4 Tensforflow model (To be implemented)
- ClickMe_test_1000.py  
    - Extract 1000 images from the original ClickMe test data
    - 1 image for 1 class, 1000 images in total
- Results_FGSM.ipynb (Results Visualization)

## results 
- FGSM.csv (to be updated)

## datasets
- clickme_test.tfrecords (Orginial ClickMe testing dataset, 17,000+ images in total)
    - please download via [link](https://drive.google.com/file/d/1-0qjq7LYGokmpXs9e6G9DbUvZxJJUAdm/view?usp=share_link)
- clickme_test_1000.tfrecords (We use this one, 1 image for 1 class, 1000 images in total)
    - please execute **ClickMe_test_1000.py**

## papers
- Model-VS-Human ([GitHub](https://github.com/bethgelab/model-vs-human/tree/745046c4d82ff884af618756bd6a5f47b6f36c45), [Paper](https://openreview.net/forum?id=QkljT4mrfs))
- Is Robustness the Cost of Accuracy? ([GitHub](https://github.com/huanzhang12/Adversarial_Survey), [Paper](https://arxiv.org/abs/1808.01688))
