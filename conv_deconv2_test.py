#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

import pdb
from time import gmtime, strftime

def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha*x,x) 

class Config:

    batch_size = 10
    num_hidden = 64
    num_epochs = 20
    l2_lambda = 0.0000001
    lr = 2e-4
    height = 256
    width = 256
    lr_D = 2e-4
    lr_G = 2e-4
    beta1_D = 0.5
    beta1_G = 0.5

class CNN_Model():

    def add_placeholders(self):
        
        self.inputs_placeholder = tf.placeholder(tf.float32, shape = [None, Config.height, Config.width, 3])
        self.targets_placeholder = tf.placeholder(tf.int32, shape = [None, Config.height, Config.width, 1])

    def create_feed_dict(self, inputs_batch, targets_batch = None):
     
        feed_dict = {}

        feed_dict[self.inputs_placeholder] = inputs_batch
        if targets_batch is not None:
            feed_dict[self.targets_placeholder] = targets_batch
        
        return feed_dict


    def add_prediction_op(self):
         
        kern_size = 4
        z = self.inputs_placeholder
        with tf.variable_scope("conv_deconv"):
            # 256 -> 128
            h1 = tf.layers.conv2d(z, filters = 64, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h1_a = leaky_relu(h1)
            print("h1: ", h1.get_shape())
            # 128 -> 64
            h2 = tf.layers.conv2d(h1_a, filters = 128, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h2 = tf.layers.batch_normalization (h2, training=True)
            h2_a = leaky_relu(h2)
            print("h2: ", h2.get_shape())
            # 64 -> 32
            h3 = tf.layers.conv2d(h2_a, filters = 256, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h3 = tf.layers.batch_normalization (h3, training=True)
            h3_a = leaky_relu(h3)
            print("h3: ", h3.get_shape())

            # 32 -> 16
            h4 = tf.layers.conv2d(h3_a, filters = 512, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h4 = tf.layers.batch_normalization (h4, training=True)
            h4_a = leaky_relu(h4)
            print("h4: ", h4.get_shape())

            # 16 -> 8
            h5 = tf.layers.conv2d(h4_a, filters = 512, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h5 = tf.layers.batch_normalization (h5, training=True)
            h5_a = leaky_relu(h5)
            print("h5: ", h5.get_shape())
            # 8 -> 4
            h6 = tf.layers.conv2d(h5_a, filters = 512, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h6 = tf.layers.batch_normalization (h6, training=True)
            h6_a = tf.nn.relu(h6)
            print("h6: ", h6.get_shape())
            # 4 -> 8
            h7 = tf.layers.conv2d_transpose(h6_a, filters = 512, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h7 = tf.layers.batch_normalization (h7, training=True)
            h7 = tf.concat([h7, h5], axis = 3) 
            h7_a = tf.nn.relu(h7)
            print("h7: ", h7.get_shape())
            # 8 -> 16
            h8 = tf.layers.conv2d_transpose(h7_a, filters = 512, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h8 = tf.layers.batch_normalization (h8, training=True)
            h8 = tf.concat([h8, h4], axis = 3) 
            h8_a = tf.nn.relu(h8)
            print("h8: ", h8.get_shape())
            # 16 -> 32
            h9 = tf.layers.conv2d_transpose(h8_a, filters = 256, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h9 = tf.layers.batch_normalization (h9, training=True)
            h9 = tf.concat([h9, h3], axis = 3) 
            h9_a = tf.nn.relu(h9)
            # 32 -> 64       
            h10 = tf.layers.conv2d_transpose(h9_a, filters = 128, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h10 = tf.layers.batch_normalization (h10, training=True)
            h10 = tf.concat([h10, h2], axis = 3)
            h10_a = tf.nn.relu(h10)
            
            # 64 -> 128
            h11 = tf.layers.conv2d_transpose(h10_a, filters = 64, kernel_size = [kern_size,kern_size], strides = [2,2], padding ='SAME')
            h11 = tf.layers.batch_normalization (h11, training=True)
            h11 = tf.concat([h11, h1], axis = 3)
            h11_a = tf.nn.relu(h11)

            # 128 -> 256    
            h12 = tf.layers.conv2d_transpose(h11_a, filters = 32, kernel_size = [kern_size,kern_size], strides = [2,2],  padding ='SAME')
            h12 = tf.layers.batch_normalization (h12, training=True)
            h12 = tf.concat([h12, z], axis = 3)
            h12_a = leaky_relu(h12, 0.1)
            
            h13 = tf.layers.conv2d(h12_a, filters = 2, kernel_size =[kern_size,kern_size], strides =[1,1], padding='SAME')
            print("h13: ", h13.get_shape())
            self.logits = h13
            # print("logits: ", self.logits.get_shape())
            self.pred = tf.argmax(input = self.logits, axis = -1) 

    def add_loss_op(self):  

        # total_loss = tf.nn.l2_loss(self.pred - self.targets_placeholder)
        #mean_loss = tf.reduce_mean(total_loss)
        onehot_labels = tf.one_hot(indices=tf.cast(self.targets_placeholder, tf.int32), depth=2)
        onehot_labels = tf.reshape(onehot_labels, [tf.shape(onehot_labels)[0], Config.height, Config.width, 2])
        # onehot_labels = tf.reshape(onehot_labels, self.logits.get_shape())
        total_loss = tf.losses.softmax_cross_entropy(\
            onehot_labels=onehot_labels, logits = self.logits)
        self.loss = total_loss

    def add_training_op(self):
    
        optimizer = tf.train.AdamOptimizer(Config.lr).minimize(self.loss)
        
        self.optimizer = optimizer

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train=True):

        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
        batch_cost = session.run(self.loss, feed)
        if math.isnan(batch_cost): # basically all examples in this batch have been skipped
            return 0
        if train:
            _ = session.run(self.optimizer, feed)

        return batch_cost / Config.batch_size

    def predict_on_batch(self, session, inputs_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch)
        predictions = session.run(self.pred, feed_dict=feed)
        return predictions


    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()

    def __init__(self):
     
        # Defining placeholders.
        self.inputs_placeholder = None
        self.labels_placeholder = None

        self.build()


if __name__ == "__main__":
    X_test = np.load('./data/images.npy')[640:,:,:,:]
    Y_test = np.load('./data/masks.npy')[640:,:,:]
    
    a,b,c,d = X_test.shape
    res = np.zeros((a,b,c1), dtype = np.uint8)

    num_examples = X_test.shape[0]
    num_batches = int(math.ceil(num_examples / Config.batch_size))

    with tf.Graph().as_default():
        model = CNN_Model() 
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.trainable_variables())
        with tf.Session() as session:
            session.run(init)
            print("loading model from checkpoint")
            #checkpoint = tf.train.latest_checkpoint("checkpoints/2017_06_09_05_59") #gan_final_gray
            checkpoint = tf.train.latest_checkpoint("checkpoints/2018_03_11_06_36") 
            saver.restore(session, checkpoint)

            for batch in range(num_batches):
                if Config.batch_size*(batch+1) < num_examples:
                    X_test_batch = X_test[Config.batch_size*batch:Config.batch_size*(batch+1)] / 255.
                else:
                    X_test_batch = X_test[Config.batch_size*batch:] / 255.            

                pred = model.predict_on_batch(session, X_test_batch)
                #np.save('res_checkpoint/pred.npy', pred)
                pred *= 255
                pred = np.uint8(pred)
                if Config.batch_size*(batch+1) < num_examples:
                    res[Config.batch_size*batch:Config.batch_size*(batch+1)] = pred
                else:
                    res[Config.batch_size*batch:] = pred
            np.save('pred_all.npy', res)