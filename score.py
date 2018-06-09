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

tf.app.flags.DEFINE_string('architecture', 'bilinear_img_1_512_0',
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data-path', "",
                           """Path where the data will be loaded.""")
args = tf.app.flags.FLAGS


def main(_):
    exp_name = 'scoring'
    log_path = './log/' + exp_name
    save_path = './model/' + exp_name
    assert args.data_path != ""
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_models = ['mysubmission']
    test_models = ['mysubmission', 'human']

    [data_train, data_val, data_test, word_embedding] = data_loader(
            args.data_path, use_mc_samples=True)
    word_to_idx = data_train['word_to_idx']

    config = Config()
    config = config_model_coco(config, args.model_architecture)
    print("Model architecture:%s"%(args.model_architecture))
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

            output_filename = '%s.txt' % (args.model_architecture)
            output_filepath = os.path.join(save_path, output_filename)
            f = open(output_filepath, 'w')

            # Training
            for i in xrange(config.max_epoch):
                print("Epoch: %d" % (i + 1))
                train_loss, train_acc = train(sess, mtrain, data_train,
                                              gen_model=train_models, epoch=i,
                                              config=config)

                for test_model in test_models:
                    [acc, logits, scores] = inference(
                            sess, mval, data_val, test_model, config=config)
                    s = np.mean(scores[:,:,0])
                    f.write("%f\t" % s)
                    a = np.mean(acc)
                    f.write("%f\t" % a)
                f.write("\n")
            f.close()

            if save_path:
                model_path = os.path.join(save_path, args.model_architecture)
                print("Saving model to %s." % model_path)
                saver.save(sess, model_path, global_step=i+1)
                print("Model saved to %s." % model_path)


if __name__ == "__main__":
    tf.app.run()
