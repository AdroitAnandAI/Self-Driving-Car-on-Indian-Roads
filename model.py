import tensorflow as tf
import scipy

def weight_variable_accel(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable_accel(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d_accel(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x_accel = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_accel_ = tf.placeholder(tf.float32, shape=[None, 1])

# isTheta = tf.placeholder(tf.bool)
x_image_accel = x_accel

#first convolutional layer
W_conv1_accel = weight_variable_accel([5, 5, 3, 24])
b_conv1_accel = bias_variable_accel([24])

h_conv1_accel = tf.nn.relu(conv2d_accel(x_image_accel, W_conv1_accel, 2) + b_conv1_accel)

#second convolutional layer
W_conv2_accel = weight_variable_accel([5, 5, 24, 36])
b_conv2_accel = bias_variable_accel([36])

h_conv2_accel = tf.nn.relu(conv2d_accel(h_conv1_accel, W_conv2_accel, 2) + b_conv2_accel)

#third convolutional layer
W_conv3_accel = weight_variable_accel([5, 5, 36, 48])
b_conv3_accel = bias_variable_accel([48])

h_conv3_accel = tf.nn.relu(conv2d_accel(h_conv2_accel, W_conv3_accel, 2) + b_conv3_accel)

#fourth convolutional layer
W_conv4_accel = weight_variable_accel([3, 3, 48, 64])
b_conv4_accel = bias_variable_accel([64])

h_conv4_accel = tf.nn.relu(conv2d_accel(h_conv3_accel, W_conv4_accel, 1) + b_conv4_accel)

#fifth convolutional layer
W_conv5_accel = weight_variable_accel([3, 3, 64, 64])
b_conv5_accel = bias_variable_accel([64])

h_conv5_accel = tf.nn.relu(conv2d_accel(h_conv4_accel, W_conv5_accel, 1) + b_conv5_accel)

#FCL 1
W_fc1_accel = weight_variable_accel([1152, 1164])
b_fc1_accel = bias_variable_accel([1164])

h_conv5_flat_accel = tf.reshape(h_conv5_accel, [-1, 1152])
h_fc1_accel = tf.nn.relu(tf.matmul(h_conv5_flat_accel, W_fc1_accel) + b_fc1_accel)

keep_prob_accel = tf.placeholder(tf.float32)
h_fc1_drop_accel = tf.nn.dropout(h_fc1_accel, keep_prob_accel)

#FCL 2
W_fc2_accel = weight_variable_accel([1164, 100])
b_fc2_accel = bias_variable_accel([100])

h_fc2_accel = tf.nn.relu(tf.matmul(h_fc1_drop_accel, W_fc2_accel) + b_fc2_accel)

h_fc2_drop_accel = tf.nn.dropout(h_fc2_accel, keep_prob_accel)

#FCL 3
W_fc3_accel = weight_variable_accel([100, 50])
b_fc3_accel = bias_variable_accel([50])

h_fc3_accel = tf.nn.relu(tf.matmul(h_fc2_drop_accel, W_fc3_accel) + b_fc3_accel)

h_fc3_drop_accel = tf.nn.dropout(h_fc3_accel, keep_prob_accel)

#FCL 3
W_fc4_accel = weight_variable_accel([50, 10])
b_fc4_accel = bias_variable_accel([10])

h_fc4_accel = tf.nn.relu(tf.matmul(h_fc3_drop_accel, W_fc4_accel) + b_fc4_accel)

h_fc4_drop_accel = tf.nn.dropout(h_fc4_accel, keep_prob_accel)

#Output
W_fc5_accel = weight_variable_accel([10, 1])
b_fc5_accel = bias_variable_accel([1])

# if (tf.cond(tf.eisTheta = True)):

# if (tf.cond(tf.equal(isTheta, tf.constant(True)), lambda: True, lambda: False)):
# if (isTheta is not None):
y_accel = tf.multiply(tf.atan(tf.matmul(h_fc4_drop_accel, W_fc5_accel) + b_fc5_accel), 2) #scale the atan output
# else:
# 	y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5 #linear output
