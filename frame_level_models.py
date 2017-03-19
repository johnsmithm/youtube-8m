# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 124, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ])

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state[0],
        vocab_size=vocab_size,
        **unused_params)

from tensorflow.contrib import grid_rnn

class LstmModel1(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a seqtoseq model to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    
    max_frames = model_input.get_shape().as_list()[1]
    batch_size = model_input.get_shape().as_list()[0]
    feature_size = model_input.get_shape().as_list()[2]
    
    self.rnn_cell = 'LSTM';
    
    if self.rnn_cell == "LSTM":
                cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    elif self.rnn_cell == "BasicLSTM":
                cell = tf.contrib.rnn.BasicLSTMCell(lstm_size,forget_bias=1.0,state_is_tuple=True)
    elif self.rnn_cell == "GRU":
                cell = tf.contrib.rnn.GRUCell(lstm_size)
    elif self.rnn_cell == "LSTMGRID2":
                cell = tf.contrib.rnn.Grid2LSTMCell(lstm_size, use_peepholes=True,forget_bias=1.0)
    elif self.rnn_cell == "LSTMGRID":
                bl = [10]
                cell = tf.contrib.rnn.GridLSTMCell(lstm_size, use_peepholes=True,forget_bias=1.0,num_frequency_blocks=bl)
    elif self.rnn_cell == "GRUGRID2":
                cell = tf.contrib.rnn.Grid2GRUCell(lstm_size)
            
    if is_training:
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.7)

    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                cell
                for _ in range(number_of_layers)
                ],
            state_is_tuple=(self.rnn_cell[-4:] == "LSTM"))

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)
    
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=(state[0][0] if self.rnn_cell[-4:] == "LSTM" else state),
        vocab_size=vocab_size,
        **unused_params)


import util_conv

class Conv3DModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a seqtoseq model to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    
    
    
    
    with tf.variable_scope("con3d"):
        model_input1 = tf.expand_dims(model_input, -1)           
        conv = {"conv0":model_input1}
        layers = 6
        for i in range(layers):
            conv["conv{}".format(i+1)] = util_conv.convLayer(conv["conv{}".format(i)],   
                                                           ((i)*2)+2, 
                                                           size_window=[2,2], 
                                                           keep_prob=None, 
                                                           maxPool=[2,2], 
                                                           scopeN="l{}".format(i))
    
        max_frames = conv["conv{}".format(layers)].get_shape().as_list()[1]
        feature_size = conv["conv{}".format(layers)].get_shape().as_list()[2]  
        out_chanels = conv["conv{}".format(layers)].get_shape().as_list()[3]  
        print(conv["conv{}".format(layers)].get_shape().as_list())
        
        flaten = tf.reshape(conv["conv{}".format(layers)],[-1, max_frames*feature_size*out_chanels])
    
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=flaten,
            vocab_size=vocab_size,
            **unused_params)
    
class Conv3DModelSlim(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a seqtoseq model to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    
    
    
    
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.expand_dims(model_input, -1)

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            #net = slim.fully_connected(net, 1024, scope='fc3')
            #net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            #outputs = slim.fully_connected(net, self.num_classes, activation_fn=None, normalizer_fn=None, scope='fco')
        
            

            aggregated_model = getattr(video_level_models,
                                       FLAGS.video_level_classifier_model)
            return aggregated_model().create_model(
                model_input=net,
                vocab_size=vocab_size,
                **unused_params)

class Seq2seq(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a seqtoseq model to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    if True:
        self.dim_image = dim_image= model_input.get_shape().as_list()[2]
        self.n_words = n_words =  vocab_size
        self.dim_hidden = dim_hidden = FLAGS.lstm_cells
        self.batch_size = tf.shape(model_input)[0]
        self.n_lstm_steps = n_lstm_steps=20
        self.drop_out_rate =drop_out_rate= 0.4
        bias_init_vector = None
        n_caption_step = 20#model_input.get_shape().as_list()[1]

        self.Wemb = tf.get_variable( 'Wemb',[n_words, dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))

        self.lstm3 = tf.contrib.rnn.LSTMCell(self.dim_hidden,
            use_peepholes = True, state_is_tuple = True)
        if is_training:
            self.lstm3_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm3,output_keep_prob=1 - self.drop_out_rate)
        else:
            self.lstm3_dropout = self.lstm3
        
        self.lstm31 = tf.contrib.rnn.LSTMCell(self.dim_hidden,
            use_peepholes = True, state_is_tuple = True)
        if is_training:
            self.lstm3_dropout1 = tf.contrib.rnn.DropoutWrapper(self.lstm31,output_keep_prob=1 - self.drop_out_rate)
        else:
            self.lstm3_dropout1 = self.lstm31
        self.encode_image_W = tf.get_variable( 'encode_image_W',[dim_image, dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))
        self.encode_image_b = tf.get_variable('encode_image_b',[dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))
        self.embed_att_w = tf.get_variable( 'embed_att_w',[dim_hidden, 1],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))
        self.embed_att_Wa = tf.get_variable( 'embed_att_Wa',[dim_hidden, dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))
        self.embed_att_Ua = tf.get_variable( 'embed_att_Ua',[dim_hidden, dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))
        self.embed_att_ba = tf.get_variable( 'embed_att_ba',[dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))

        self.embed_word_W = tf.get_variable('embed_word_W',[dim_hidden, n_words],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(n_words)))
        if bias_init_vector is not None:
             self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.get_variable( 'embed_word_b',[n_words],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(n_words)))

        self.embed_nn_Wp = tf.get_variable( 'embed_nn_Wp',[3*dim_hidden, dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))
        self.embed_nn_bp = tf.get_variable('embed_nn_bp',[dim_hidden],
                                          initializer = 
                                           tf.random_normal_initializer(stddev=1 / math.sqrt(dim_hidden)))
        
        #print(model_input.get_shape().as_list())
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        video = utils.SampleRandomFrames(model_input, num_frames, n_lstm_steps)
        #print(video.get_shape().as_list())
        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b x n x h
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h

        state1 = self.lstm3.zero_state(self.batch_size, dtype=tf.float32)#tf.zeros([self.batch_size, self.lstm3.state_size]) # b x s
        h_prev = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        
        state11 = self.lstm31.zero_state(self.batch_size, dtype=tf.float32)# # b x s
        h_prev1 = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        loss_caption = 0.0
        
        probs = []

        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        
        image_part = tf.reshape(image_emb, [-1, self.dim_hidden])
        image_part = tf.matmul(image_part, self.embed_att_Ua) + self.embed_att_ba
        image_part = tf.reshape(image_part, [self.n_lstm_steps, self.batch_size, self.dim_hidden])
        with tf.variable_scope("model") as scope:
            for i in range(n_caption_step):
                e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
    #            e = tf.batch_matmul(e, brcst_w)    # unnormalized relevance score 
                e = tf.reshape(e, [-1, self.dim_hidden])
                e = tf.matmul(e, self.embed_att_w) # n x b
                e = tf.reshape(e, [self.n_lstm_steps, self.batch_size])
    #            e = tf.reduce_sum(e,2) # n x b
                e_hat_exp = tf.exp(e)#tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b 
                denomin = tf.reduce_sum(e_hat_exp,0) # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
                alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
                attention_list = tf.multiply(alphas, image_emb) # n x b x h
                atten = tf.reduce_sum(attention_list,0) # b x h       #  soft-attention weighted sum
#                if i > 0: tf.get_variable_scope().reuse_variables()
                if i > 0: scope.reuse_variables()

                with tf.variable_scope("LSTM3"):
                    output12, state11 = self.lstm3_dropout1(tf.concat([atten, current_embed], 1), state11 ) # b x h
                with tf.variable_scope("LSTM31"):    
                    output1, state1 = self.lstm3_dropout(output12, state1 ) # b x h

                output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1,atten,current_embed], 1), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
                h_prev = output1 # b x h               

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) # b x w
                probs.append(logit_words)
                        
        tf_probs = tf.stack(probs,0)
        tf_probs = tf.transpose(tf_probs,[1,0,2])
        return { 'predictions': tf.nn.softmax(tf.reduce_mean(tf_probs,1))    }
