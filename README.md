# MonoGID

## Overview

MonoGID is a self-supervised monocular endoscopic depth estimation framework designed to improve geometric representation and robustness under challenging endoscopic imaging conditions, such as weak textures, specular reflections, shadows, and local overexposure.

This repository provides the code for training and evaluation on the **SCARED** dataset, as well as pretrained model weights for inference and reproduction of the reported results.

## Dataset

### SCARED

Please follow the dataset preparation procedure used in [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) to prepare the SCARED dataset.

## Usage

Download pretrained model from: [depth_anything_vitb14](https://drive.google.com/file/d/1e9Hmy2PjE1EBu-W44zJ5CfbrA-xH5gQt/view?usp=sharing). Create a folder named `pretrained_model` in this repo and place the downloaded model in it.

### Evaluation

Before evaluation, export the ground-truth depth:

```bash
CUDA_VISIBLE_DEVICES=0 python export_gt_depth.py --data_path <your_data_path> --split endovis
```

Then evaluate the model. For example, to evaluate the weights from epoch 19:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --load_weights_folder './logs/models/weights_19' --eval_mono
```

## Model Checkpoints

The model checkpoints used to reproduce the results reported in the paper are available at the following Google Drive link:

[Google Drive](https://drive.google.com/file/d/1nGtdylHEK0nuYnABy81URXZkdnnVxQGT/view?usp=sharing)
