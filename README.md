# TF - Fashion Domain Transfer Network

### Introduction

This is TensorFlow version of [Unsupervised Cross-Domain Image Generation (2016)](https://arxiv.org/abs/1611.02200), which transfers Fashion-MNIST images to real clothes images.

### Paper Summary

Will be added...

![](./images/summary.png)

![](./images/loss.png)

### Requirements

- Python >= 3.5
- TensorFlow >= 1.4
- python-opencv (cv2)
- argparse
- tqdm

### Install repository

1. Clone this repository

`git clone https://github.com/MagmaTart/TF-Domain-Transfer-Network.git`

2. Move to repository (On machine local)

`cd TF-Domain-Transfer-Network`

3. Download Fashion-MNIST dataset and move to repository. Do initialize settings.

`./setting.sh`

4. Download Clothes image dataset on [this link](https://www.dropbox.com/sh/ryl8efwispnjw21/AACt2dLasqSDsCf-kcQwoWyfa?dl=0). You should download `Anno/list_bbox.txt` and `Img/img.zip`

5. Move Clothes dataset to repository

`cd TF-Domain-Transfer-Network`

`mv ~/Downloads/list_bbox.txt ~/Downloads/img.zip Fashion-images`

`unzip img.zip`

`mv img/* .`

`rm img.zip`

### Setting and Test Codes

You should work on TensorFlow environment, and this repository.

1. Do preprocess to test

`python main.py --mode preprocessing`

2. Pretrain the Feature extractor model

`python main.py --mode pretrain`

3. Test pretrained model

`python main.py --mode pretrain-test`

4. Train the model

`python main.py --mode train`

__...Under Construction...__

### TO-DOs

- Fix model import problem when test process
- Make evaluation functions
- Fix mode collapse problem
- Fix blurry image problem (I tried : L1, L2, Hinge loss)

### Current result

Will be added...

### Memo

- It is important to well train feature extractor model
