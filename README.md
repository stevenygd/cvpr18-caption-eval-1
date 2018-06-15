# CVPR 2018 Captioning Evaluation
TensorFlow implementation for [*Learning to Evaluate Image Captioning (CVPR 2018)*](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/1501.pdf)

Authors: Yin Cui, Guandao Yang, Andreas Veit, Xun Huang, Serge Belongie


## Introduction

This repository contains a discriminator that could be trained to evaluate image captioning systems. The discriminator is trained to distinguish between machine generated captions and human written ones. During testing, the trained discriminator take the cadidate caption, the reference caption, and optionally the image to be captioned as input. Its output probability of how likely the candidate caption is human written can be used to evaluate the candidate caption. Please refer to our paper [[link]](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/1501.pdf) for more detail.

<p align="center">
  <img src="figs/TrainingDiagram.png" width="100%">
</p>


If you find our work helpful in your research, please cite it as:
```latex
@inproceedings{Cui2018CaptionEval,
  title = {Learning to Evaluate Image Captioning},
  author = {Yin Cui, Guandao Yang, Andreas Veit, Xun Huang, and Serge Belongie},
  booktitle={CVPR},
  year={2018}
}
```


## Dependencies

+ Python (2.7)
+ Tensorflow (>1.4)
+ OpenCV
+ PyTorch (for extracting ResNet image features.)
+ ProgressBar
+ NLTK

## Preparation

1. Clone the dataset with recursive (include the bilinear pooling)
```bash
git clone --recursive https://github.com/richardaecn/cvpr18-caption-eval.git
```
2. Install dependencies. Please refer to TensorFlow, PyTorch, NTLK, and OpenCV's official websites for installation guide. For other dependencies, please use the following:
```bash
pip install -r requirements.txt
```

3. Download data. This script will download data for both robustness and capability experiments, data created by Monte Carlo sampling, Karpathy's split, Glove embedding, and MSCOCO images.
```bash
./download.sh
```

4. Generate vocabulrary.
```bash
python scripts/preparation/prep_vocab.py
```

5. Extract image features. Following script will download ResNet checkpoint and use the checkpoint to extract the image features from MSCOCO dataset. This might take up to five hours.
```bash
cd scripts/features/
./download.sh
python feature_extraction_coco.py --data-dir ../../data/ --coco-img-dir ../../data
```


## Evaluate A Single Submission

To evaluate a single submission, first put the output captions of a model into the following JSON format:

```json
{
    "<file-name-1>" : "<caption-1>",
    "<file-name-2>" : "<caption-2>",
    ...
    "<file-name-n>" : "<caption-n>",
}
```

Note that ```<caption-i>``` are caption represented in text, and the file name is the name for the file in the image. The caption should be all lower-cased and have no ```\n``` at the end. Examples of such file can be found in the ```examples``` folder: ```examples/neuraltalk_all_captions.json```, ```examples/showandtell_all_captions.json```, ```examples/showattendandtell_all_captions.json```, and ```examples/human_all_captions.json```.
Following command prepared the data so that it could be used for training:

```bash
python scripts/preparation/prep_submission.py --submission examples/neuraltalk_all_captions.json  --name neuraltalk
```

Note that we assume you've followed through the steps in the *Preparation* section before running this command. This script will create a folder `data/neuraltalk` and three ```.npy``` files that contain data needed for training the metric. Please use the following command to train the metric:

```bash
python score.py --name neuraltalk
```

The results will be logged in `model/neuraltalk_scoring` directory. If you use the default model architecture, the results will be in `model/neuraltalk_scoring/bilinear_img_1_512_0.txt`.

Followings are the score for the submissions. (Scores might be slightly different due to randomization.)

| Architecture         | Epochs | Neuraltalk | Showandtell | Showattendtell | Human |
|----------------------|--------|------------|-------------|----------------|-------|
| bilinear_img_1_512_0 | 10     | 0.079      | 0.103       | 0.121          | 0.605 |
| bilinear_img_1_512_0 | 30     | 0.066      | 0.073       | 0.111          | 0.611 |

