#! /bin/bash

#   1. Glove 6B (from offical website, move to ./glove )
wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir glove
mv glove.6B.zip glove
cd glove
unzip glove.6B.zip
rm glove.6B.zip
cd ..

#   2. Download the kaparthy split (dataset_coco.json, move to ./data/karpathysplit/dataset_coco.json)
#   2. Alpha/Beta/Gemma scores (scores_alpha.npy, scores_beta.npy, scores_gamma.npy, move to ./data)
#   3. MC Sampled Captions (dumped_train.npy, dumped_val.npy, dumped_test.npy, move to ./data)
#   4. preprocessed data for the experiments (data_train_full.npy, data_val_full.npy, data_test_full.npy)
