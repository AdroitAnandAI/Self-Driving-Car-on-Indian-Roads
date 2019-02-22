
## This model is used to predict Acceleration

import tensorflow as tf
import scipy

def weight_variable_brake(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable_brake(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d_brake(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x_brake = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_brake_ = tf.placeholder(tf.float32, shape=[None, 1])

# isTheta = tf.placeholder(tf.bool)
x_image_brake = x_brake

#first convolutional layer
W_conv1_brake = weight_variable_brake([5, 5, 3, 24])
b_conv1_brake = bias_variable_brake([24])

h_conv1_brake = tf.nn.relu(conv2d_brake(x_image_brake, W_conv1_brake, 2) + b_conv1_brake)

#second convolutional layer
W_conv2_brake = weight_variable_brake([5, 5, 24, 36])
b_conv2_brake = bias_variable_brake([36])

h_conv2_brake = tf.nn.relu(conv2d_brake(h_conv1_brake, W_conv2_brake, 2) + b_conv2_brake)

#third convolutional layer
W_conv3_brake = weight_variable_brake([5, 5, 36, 48])
b_conv3_brake = bias_variable_brake([48])

h_conv3_brake = tf.nn.relu(conv2d_brake(h_conv2_brake, W_conv3_brake, 2) + b_conv3_brake)

#fourth convolutional layer
W_conv4_brake = weight_variable_brake([3, 3, 48, 64])
b_conv4_brake = bias_variable_brake([64])

h_conv4_brake = tf.nn.relu(conv2d_brake(h_conv3_brake, W_conv4_brake, 1) + b_conv4_brake)

#fifth convolutional layer
W_conv5_brake = weight_variable_brake([3, 3, 64, 64])
b_conv5_brake = bias_variable_brake([64])

h_conv5_brake = tf.nn.relu(conv2d_brake(h_conv4_brake, W_conv5_brake, 1) + b_conv5_brake)

#FCL 1
W_fc1_brake = weight_variable_brake([1152, 1164])
b_fc1_brake = bias_variable_brake([1164])

h_conv5_flat_brake = tf.reshape(h_conv5_brake, [-1, 1152])
h_fc1_brake = tf.nn.relu(tf.matmul(h_conv5_flat_brake, W_fc1_brake) + b_fc1_brake)

keep_prob_brake = tf.placeholder(tf.float32)
h_fc1_drop_brake = tf.nn.dropout(h_fc1_brake, keep_prob_brake)

#FCL 2
W_fc2_brake = weight_variable_brake([1164, 100])
b_fc2_brake = bias_variable_brake([100])

h_fc2_brake = tf.nn.relu(tf.matmul(h_fc1_drop_brake, W_fc2_brake) + b_fc2_brake)

h_fc2_drop_brake = tf.nn.dropout(h_fc2_brake, keep_prob_brake)

#FCL 3
W_fc3_brake = weight_variable_brake([100, 50])
b_fc3_brake = bias_variable_brake([50])

h_fc3_brake = tf.nn.relu(tf.matmul(h_fc2_drop_brake, W_fc3_brake) + b_fc3_brake)

h_fc3_drop_brake = tf.nn.dropout(h_fc3_brake, keep_prob_brake)

#FCL 3
W_fc4_brake = weight_variable_brake([50, 10])
b_fc4_brake = bias_variable_brake([10])

h_fc4_brake = tf.nn.relu(tf.matmul(h_fc3_drop_brake, W_fc4_brake) + b_fc4_brake)

h_fc4_drop_brake = tf.nn.dropout(h_fc4_brake, keep_prob_brake)

#Output
W_fc5_brake = weight_variable_brake([10, 1])
b_fc5_brake = bias_variable_brake([1])

# if (tf.cond(tf.eisTheta = True)):

# if (tf.cond(tf.equal(isTheta, tf.constant(True)), lambda: True, lambda: False)):
# if (isTheta is not None):
y_brake = tf.multiply(tf.atan(tf.matmul(h_fc4_drop_brake, W_fc5_brake) + b_fc5_brake), 2) #scale the atan output
# else:
# 	y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5 #linear output
