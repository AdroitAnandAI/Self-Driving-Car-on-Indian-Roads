import tensorflow as tf
import scipy

def weight_variable_brake(shape):
  
  # initializer = tf.contrib.layers.variance_scaling_initializer()
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape))

  # initial = tf.truncated_normal(shape, stddev=0.1)
  # return tf.Variable(initial)

def bias_variable_brake(shape):

  # initializer = tf.contrib.layers.variance_scaling_initializer()
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape))


  # initial = tf.constant(0.1, shape=shape)
  # return tf.Variable(initial)


def conv2d_brake(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

x_brake = tf.placeholder(tf.float32, shape=[None, 112, 112, 3])
y_brake_ = tf.placeholder(tf.float32, shape=[None, 1])

# isTheta = tf.placeholder(tf.bool)
x_image_brake = x_brake

# Block 1
#first convolutional layer
W_conv1_brake = weight_variable_brake([3, 3, 3, 256])
b_conv1_brake = bias_variable_brake([256])
h_conv1_brake = tf.nn.relu(conv2d_brake(x_image_brake, W_conv1_brake, 1) + b_conv1_brake)

h_conv1_brake_pool = tf.nn.max_pool(h_conv1_brake, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')


keep_prob_brake_conv = tf.placeholder(tf.float32)
h_conv1_drop_brake = tf.nn.dropout(h_conv1_brake_pool, keep_prob_brake_conv)

#second convolutional layer
W_conv2_brake = weight_variable_brake([3, 3, 256, 128])
b_conv2_brake = bias_variable_brake([128])
h_conv2_brake = tf.nn.relu(conv2d_brake(h_conv1_drop_brake, W_conv2_brake, 1) + b_conv2_brake)

h_conv2_brake_pool = tf.nn.max_pool(h_conv2_brake, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
h_conv2_drop_brake = tf.nn.dropout(h_conv2_brake_pool, keep_prob_brake_conv)

print(h_conv2_drop_brake.shape)

#FCL 1
W_fc1_brake = weight_variable_brake([28*28*128, 128])
b_fc1_brake = bias_variable_brake([128])

# print(h_conv4_brake_pool.shape)
h_conv2_flat_brake = tf.reshape(h_conv2_drop_brake, [-1, 28*28*128])
h_fc1_brake = tf.nn.relu(tf.matmul(h_conv2_flat_brake, W_fc1_brake) + b_fc1_brake)

keep_prob_brake = tf.placeholder(tf.float32)
h_fc1_drop_brake = tf.nn.dropout(h_fc1_brake, keep_prob_brake)


#FCL 2
W_fc2_brake = weight_variable_brake([128, 64])
b_fc2_brake = bias_variable_brake([64])

h_fc2_brake = tf.nn.relu(tf.matmul(h_fc1_drop_brake, W_fc2_brake) + b_fc2_brake)
h_fc2_drop_brake = tf.nn.dropout(h_fc2_brake, keep_prob_brake)

#FCL 3
W_fc3_brake = weight_variable_brake([64, 1])
b_fc3_brake = bias_variable_brake([1])

y_brake = tf.matmul(h_fc2_drop_brake, W_fc3_brake) + b_fc3_brake #linear output
