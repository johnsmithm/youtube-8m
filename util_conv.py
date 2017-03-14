import tensorflow as tf
import math

def weight_variable(shape, name="v"):
    return tf.get_variable(name+"_weight",
      shape,
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(shape[3])))

def bias_variable(shape, name="v"):
    return tf.get_variable(name+"_bias",
      shape,
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(4)))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, h=2, w=2):
  return tf.nn.max_pool(x, ksize=[1, h, w, 1],
                        strides=[1, h, w, 1], padding='SAME')


def convLayer(data, chanels_out, size_window, keep_prob=0.8, maxPool=None, scopeN="l1"):
    """Implement convolutional layer
    @param data: [batch,h,w,chanels]
    @param chanels_out: number of out chanels
    @param size_windows: the windows size
    @param keep_prob: the dropout amount
    @param maxPool: if true the max pool is applyed
    @param scopeN: the scope name
    
    returns convolutional output [batch,h,w,chanels_out]
    """
    with tf.name_scope("conv-"+scopeN):
        shape = data.get_shape().as_list()
        with tf.variable_scope("convVars-"+scopeN) as scope:
            W_conv1 = weight_variable([size_window[0], size_window[1], shape[3], chanels_out], scopeN)
            b_conv1 = bias_variable([chanels_out], scopeN)
        h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
        if keep_prob and keep_prob!=1 and self.train_b:
            h_conv1 = tf.nn.dropout(h_conv1, keep_prob)
        if maxPool is not None:
            h_conv1 = max_pool_2x2(h_conv1, maxPool[0], maxPool[1])
    return h_conv1


