"""
To run:

$ python discriminator.py'
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os
import random
import copy

from tensorflow.python.util import nest
from discriminator import *
from config import *

def inference(sess, model, data, gen_model, robustness_idx=0, dim_feat=2048, config=Config()):
    """Runs the model on the given data."""
    num_steps = config.num_steps
    num_input = config.num_input
    batch_size = config.batch_size
    if "file_names" in data:
        filename = data['file_names']
    else:
        filename = data['image_ids']
    acc = []
    logits = []
    scores = []

    idx = range(len(filename))
    epoch_size = len(idx) // batch_size
    if batch_size * epoch_size < len(idx):
        epoch_size += 1
        idx.extend(idx[:batch_size * epoch_size - len(idx)])

    for i in xrange(epoch_size):
        if i == epoch_size - 1:
            idx_batch = idx[batch_size*i:]
        else:
            idx_batch = idx[batch_size*i:batch_size*(i+1)]

        x = np.zeros((len(idx_batch)*num_input, num_steps), dtype=np.int32)
        y_ = np.zeros((len(idx_batch), 2), dtype=np.float32)
        y_[:, 1] = 1.0
        img = np.zeros((len(idx_batch)*num_input, dim_feat), dtype=np.float32)

        for j in xrange(len(idx_batch)):
            img_feat = copy.deepcopy(data['features'][filename[idx_batch[j]]])
            real_cap = copy.deepcopy(
                    data['captions'][filename[idx_batch[j]]]['human_baseline_gt'])
            real_idx = range(len(real_cap))
            random.shuffle(real_idx)
            x[j, :] = real_cap[real_idx[0]]
            img[j,:] = img_feat

            x[j+len(idx_batch), :] = copy.deepcopy(
                    data['captions'][filename[idx_batch[j]]][gen_model][robustness_idx])
            img[j+len(idx_batch),:] = img_feat

        acc_batch, logits_batch, scores_batch = sess.run(
                [model._accuracy, model._logits, model._score],
                {model.x: x, model.y_: y_, model.img_feat:img}
            )
        acc.append(acc_batch)
        logits.append(logits_batch)
        scores.append(scores_batch)

    print('%s Average Score: %.3f' % (gen_model, np.mean(np.array(scores)[:,:,0])))
    return np.array(acc), np.array(logits), np.array(scores)


def main(_):
    exp_name = 'robustness'
    data_path = './data'
    log_path = './log/' + exp_name
    save_path = './model/' + exp_name
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_architectures = [
        'bilinear_img_1_512_0',
        'concat_img_1_512_0',
        'mlp_1_img_1_512_0',
        'bilinear_img_1_512_0_noda',
        'concat_img_1_512_0_noda',
        'mlp_1_img_1_512_0_noda'
    ]
    train_models = ['neuraltalk', 'showandtell', 'showattendtell']
    test_models = ['alpha', 'beta', 'gamma']

    [data_train, data_val, data_test, word_embedding] = data_loader(
            data_path, use_mc_samples=True)
    word_to_idx = data_train['word_to_idx']

    data = np.load('./data/data_robustness.npy').item()

    for model_architecture in model_architectures:
        config = Config()
        config = config_model_coco(config, model_architecture)
        with tf.Graph().as_default():
            with tf.name_scope("Train"):
                with tf.variable_scope("Discriminator", reuse=None):
                    mtrain = Discriminator(word_embedding, word_to_idx, use_glove=True,
                                           config=config, is_training=True)
                tf.summary.scalar("Training Loss", mtrain._loss)
                tf.summary.scalar("Training Accuracy", mtrain._accuracy)

            with tf.name_scope("Val"):
                with tf.variable_scope("Discriminator", reuse=True):
                    mval = Discriminator(word_embedding, word_to_idx, use_glove=True,
                                         config=config, is_training=False)
                tf.summary.scalar("Validation Loss", mval._loss)
                tf.summary.scalar("Validation Accuracy", mval._accuracy)

            config_sess = tf.ConfigProto(allow_soft_placement=True)
            config_sess.gpu_options.allow_growth = True
            with tf.Session(config=config_sess) as sess:
                tf.global_variables_initializer().run()
                summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
                saver = tf.train.Saver()

                for i in xrange(config.max_epoch):
                    print("Epoch: %d" % (i + 1))
                    print(len(train_models))
                    train_loss, train_acc = train(sess, mtrain, data_train, train_models,
                                                  i, config=config)

                for test_model in test_models:
                    output_filename = '%s_%s_robustness.txt'% (model_architecture, test_model)
                    output_filepath = os.path.join(save_path, output_filename)

                    f = open(output_filepath, 'w')
                    for i in xrange(21):
                        [acc, logits, scores] = inference(sess, mval, data, test_model,
                                                          robustness_idx=i, config=config)
                        s = np.mean(scores[:,:,0])
                        print i, s
                        f.write("%f " % s)
                    f.close()

                if save_path:
                    model_path = os.path.join(save_path, model_architecture)
                    print("Saving model to %s." % model_path)
                    saver.save(sess, model_path, global_step=i+1)
                    print("Model saved to %s." % model_path)


if __name__ == "__main__":
    tf.app.run()
