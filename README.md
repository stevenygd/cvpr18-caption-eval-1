# CVPR 2018 Captioning Evaluation
TensorFlow implementation for paper [*Learning to Evaluate Image Captioning (CVPR 2018)*](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/1501.pdf)

Authors: Yin Cui, Guandao Yang, Andreas Veit, Xun Huang, Serge Belongie


## Introduction

This repository contains a discriminator that could be trained to evaluate image captioning systems. The discriminator is trained to distinguish between machine generated captions and human written ones. During testing time, the trained discriminator take the image to be captioned, the reference caption, and the candidate caption. Its output probability of how likely the candidate caption is human written can be used to evaluate how well the candidate caption is. Please refer to the paper [[link]](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/1501.pdf) for more detail.

<p align="center">
  <img src="figs/TrainingDiagram.png" width="100%">
</p>


If you find our work helpful in your research, please cite it:
```latex
@conference{Cui2018CaptionEval,
title = {Learning to Evaluate Image Captioning},
author = {Yin Cui, Guandao Yang, Andreas Veit, Xun Huang, and Serge Belongie},
url = {https://vision.cornell.edu/se3/wp-content/uploads/2018/03/1501.pdf},
year = {2018},
date = {2018-06-18},
booktitle = {Computer Vision and Pattern Recognition (CVPR)},
address = {Salt Lake City, UT},
keywords = {}
}
```


## Dependencies

+ Python (2.7)
+ Tensorflow (>1.4)
+ OpenCV

## Preparation

1. Clone the dataset with recursive (include the bilinear pooling)

2. Install dependency (1. tf1.4, opencv, compact-bilinear-pooling)

3. Download data (MC-samplers, vocab, alpha/beta/gemma scores for other metrics)

4. Preprocessing Image (image resize, extract image feature)

5. Introduce how to run each of the experiments
    (NOTE: if they want to do that, need to download the data for the three models we trained on)

## Running Experiments

1. Capability experiments
```bash
python discriminator_capability.py
```

2. Robustness experiments
```bash
python discriminator_robustness.py
```

## Results

To observe the result, please use the jupyter notebook ```plots.ipynb```. 

1. Capability Experiment
<p align="center">
  <img src="figs/turing_test_human.png" width=400>
</p>
<p align="center">
  <img src="figs/turing_test_gen.png" width=400>
</p>


2. Robustness Experiment
<p align="center">
  <img src="figs/alpha_ours.png" width=250>
  <img src="figs/beta_ours.png" width=250>
  <img src="figs/gamma_ours.png" width=250>
 </p>
<p align="center">
  <img src="figs/alpha_all.png" width=250>
  <img src="figs/beta_all.png" width=250>
  <img src="figs/gamma_all.png" width=250>
</p>


## Evaluate a Model

To use our metric, first put the output captions of a model into following JSON format:

```json
{
    "<file-name>": [
        "<caption-1>",
        "<caption-2>",
        ...,
        "<caption-n>"
    ]
}
```

Note that ```<caption-n>``` are caption represented in text, and the file name is the name for the file in the image. Suppose the json file has name ```submission.json```. Use the following command to get the score:

```
python gen_data.py submission.json dict.json output.npy # tokenize, put into the dsire format
python score.py --train-model-data output.npy
```
