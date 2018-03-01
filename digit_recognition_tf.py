# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:29:06 2017

@author: Anil Marella
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

LOGDIR = 'logdir/'
DATA_DIR = 'data/'
mnist = mnist_data.read_data_sets(DATA_DIR, one_hot=True, reshape=False, validation_size=0)

def conv_layer(input, filter_size, stride, channels_in, channels_out, name="convolutional_layer"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([filter_size[0], filter_size[1], channels_in, channels_out], stddev=0.1), name = "W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
        act = tf.nn.relu(tf.nn.conv2d(input, w, strides=[1,stride[0], stride[1], 1], padding='SAME')+b)
        tf.summary.histogram("Weights",w)
        tf.summary.histogram("Biases",b)    
        tf.summary.histogram("Activation",act)
        return act

def fc_layer(input, channels_in, channels_out, pkeep, name="fully_connected_layer"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape = [channels_out]), name="B")
        if(name == "output_layer"):
            act = tf.nn.softmax(tf.matmul(input, w) + b)
        else:
            act = tf.nn.relu(tf.matmul(input, w) + b)
            act = tf.nn.dropout(act, pkeep)
        tf.summary.histogram("Weights",w)
        tf.summary.histogram("Biases",b)    
        tf.summary.histogram("Activation",act)
        return act

def mnist_model(x, pkeep=1.0):
    # Check if images look ok.
    tf.summary.image('input', x, 3)
    
    # Initialize layers
    # Convolutional layers
    conv_layer1 = conv_layer(x, [6, 6],[1,1], 1, 6, "convolutional_1")
    conv_layer2 = conv_layer(conv_layer1, [5, 5],[2,2], 6, 12, "convolutional_2")
    conv_layer3 = conv_layer(conv_layer2, [4, 4],[2,2], 12, 24, "convolutional_3")
    
    # Flattened fully-connected layer
    flattened = tf.reshape(conv_layer3, shape=[-1,7*7*24])
    fc_hidden_layer1 = fc_layer(flattened, 7*7*24, 200, pkeep, "flattened_layer")
    logits = fc_layer(fc_hidden_layer1, 200, 10, pkeep, "output_layer")
    
    return logits
    
def calc_cost(logits, y):
    with tf.name_scope("Cross_Entropy_Cost"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))*100
        tf.summary.scalar("cross_entropy", cross_entropy)
    with tf.name_scope("Accuracy_measure"):
        is_correct = tf.equal(tf.arg_max(logits,1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    return cross_entropy, accuracy

def main():
    # Create placeholders
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
    y = tf.placeholder(tf.int32, [None, 10], name="labels")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    pkeep = tf.placeholder(tf.float32, name="dropout_prob")    
    
    cross_entropy, accuracy = calc_cost(mnist_model(x, pkeep), y)
    
    with tf.name_scope("Cost_Optimization"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)        

    summaries = tf.summary.merge_all()
        
    # Init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(LOGDIR+"train")
    test_writer = tf.summary.FileWriter(LOGDIR+"test")
    
    # Add graph to the writer
    train_writer.add_graph(sess.graph)
     
    num_epochs = 20
    epoch = 0
    batch_size = 100
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    loopCounter = 0
    # Training the model
    
    while epoch <= num_epochs:
        epoch = mnist.train.epochs_completed
        # learning rate decay
        lr = max_learning_rate - ((max_learning_rate - min_learning_rate) * (epoch/(num_epochs-1)))
        
        # Select batches of batch_size
        x_batch, y_batch = mnist.train.next_batch(batch_size)

        if loopCounter % 10 == 0:
            train_summ = sess.run(summaries, feed_dict={x: x_batch, y: y_batch, learning_rate: lr, pkeep: 0.75})
            train_writer.add_summary(train_summ, loopCounter)
        if loopCounter % 100 == 0:
            test_summ = sess.run(summaries, feed_dict={x: mnist.test.images, y: mnist.test.labels, pkeep:1.0})
            test_writer.add_summary(test_summ, loopCounter)
        
        sess.run(train_step, feed_dict={x: x_batch, y: y_batch, learning_rate: lr, pkeep: 0.75})
        loopCounter += 1
    sess.close()

if __name__ == '__main__':
  main()