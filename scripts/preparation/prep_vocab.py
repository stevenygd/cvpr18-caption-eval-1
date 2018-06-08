import json
import argparse
import time
import os
import numpy as np
import progressbar

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--split', type=str, default="./data/karpathysplit/dataset_coco.json",
        help='Path to the JSON file that contains how to split the dataset')
parser.add_argument(
        '--output-path', type=str, default="./data",
        help='Path to the JSON file that contains how to split the dataset')

args = parser.parse_args()

# Load splits, default to be Kaparthy's split
split = json.loads(open(args.split).read())
coco_dict = {}
token_len = []
pbar = progressbar.ProgressBar()
for i in pbar(range(len(split['images']))):
    if split['images'][i]['split'] == 'train' or split['images'][i]['split'] == 'restval':
        anns = split['images'][i]['sentences']
        for j in range(len(anns)):
            tokens = anns[j]['tokens']
            token_len.append(len(tokens))
            for k in range(len(tokens)):
                token = tokens[k].lower()
                if token in coco_dict:
                    coco_dict[token] += 1
                else:
                    coco_dict[token] = 1

import operator
sorted_dict = sorted(coco_dict.items(), key=operator.itemgetter(1))
sorted_dict = sorted_dict[::-1]

coco_word_10k = []
coco_word_10k.append('<pad>')
coco_word_10k.append('<unk>')
coco_word_10k.append('<sos>')
coco_word_10k.append('<eos>')
for i in range(10000):
    coco_word_10k.append(sorted_dict[i][0])
print("Vocab length:%d"%len(coco_word_10k))

word_to_idx = {}
for i in range(len(coco_word_10k)):
    word_to_idx[coco_word_10k[i]] = i

np.save(os.path.join(args.output_path, 'word_to_idx.npy'),  word_to_idx)
