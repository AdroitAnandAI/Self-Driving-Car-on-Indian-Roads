import tensorflow as tf
import scipy

def weight_variable_accel(shape):
  
  # initializer = tf.contrib.layers.variance_scaling_initializer()
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape))

  # initial = tf.truncated_normal(shape, stddev=0.1)
  # return tf.Variable(initial)

def bias_variable_accel(shape):

  # initializer = tf.contrib.layers.variance_scaling_initializer()
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape))


  # initial = tf.constant(0.1, shape=shape)
  # return tf.Variable(initial)


def conv2d_accel(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

x_accel = tf.placeholder(tf.float32, shape=[None, 112, 112, 3])
y_accel_ = tf.placeholder(tf.float32, shape=[None, 1])

# isTheta = tf.placeholder(tf.bool)
x_image_accel = x_accel

# Block 1
#first convolutional layer
W_conv1_accel = weight_variable_accel([3, 3, 3, 256])
b_conv1_accel = bias_variable_accel([256])
h_conv1_accel = tf.nn.relu(conv2d_accel(x_image_accel, W_conv1_accel, 1) + b_conv1_accel)

h_conv1_accel_pool = tf.nn.max_pool(h_conv1_accel, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')


keep_prob_accel_conv = tf.placeholder(tf.float32)
h_conv1_drop_accel = tf.nn.dropout(h_conv1_accel_pool, keep_prob_accel_conv)

#second convolutional layer
W_conv2_accel = weight_variable_accel([3, 3, 256, 128])
b_conv2_accel = bias_variable_accel([128])
h_conv2_accel = tf.nn.relu(conv2d_accel(h_conv1_drop_accel, W_conv2_accel, 1) + b_conv2_accel)

h_conv2_accel_pool = tf.nn.max_pool(h_conv2_accel, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
h_conv2_drop_accel = tf.nn.dropout(h_conv2_accel_pool, keep_prob_accel_conv)

print(h_conv2_drop_accel.shape)

#FCL 1
W_fc1_accel = weight_variable_accel([28*28*128, 128])
b_fc1_accel = bias_variable_accel([128])

# print(h_conv4_accel_pool.shape)
h_conv2_flat_accel = tf.reshape(h_conv2_drop_accel, [-1, 28*28*128])
h_fc1_accel = tf.nn.relu(tf.matmul(h_conv2_flat_accel, W_fc1_accel) + b_fc1_accel)

keep_prob_accel = tf.placeholder(tf.float32)
h_fc1_drop_accel = tf.nn.dropout(h_fc1_accel, keep_prob_accel)


#FCL 2
W_fc2_accel = weight_variable_accel([128, 64])
b_fc2_accel = bias_variable_accel([64])

h_fc2_accel = tf.nn.relu(tf.matmul(h_fc1_drop_accel, W_fc2_accel) + b_fc2_accel)
h_fc2_drop_accel = tf.nn.dropout(h_fc2_accel, keep_prob_accel)

#FCL 3
W_fc3_accel = weight_variable_accel([64, 1])
b_fc3_accel = bias_variable_accel([1])

y_accel = tf.matmul(h_fc2_drop_accel, W_fc3_accel) + b_fc3_accel #linear output
