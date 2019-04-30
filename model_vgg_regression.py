import tensorflow as tf
import scipy

def weight_variable_accel(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable_accel(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d_accel(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

x_accel = tf.placeholder(tf.float32, shape=[None, 112, 112, 3])
y_accel_ = tf.placeholder(tf.float32, shape=[None, 1])

# isTheta = tf.placeholder(tf.bool)
x_image_accel = x_accel

# Block 1
#first convolutional layer
W_conv1_accel = weight_variable_accel([3, 3, 3, 64])
b_conv1_accel = bias_variable_accel([64])
h_conv1_accel = tf.nn.relu(conv2d_accel(x_image_accel, W_conv1_accel, 1) + b_conv1_accel)

#second convolutional layer
W_conv2_accel = weight_variable_accel([3, 3, 64, 64])
b_conv2_accel = bias_variable_accel([64])
h_conv2_accel = tf.nn.relu(conv2d_accel(h_conv1_accel, W_conv2_accel, 1) + b_conv2_accel)

h_conv2_accel_pool = tf.nn.max_pool(h_conv2_accel, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')



# Block 2
#third convolutional layer
W_conv3_accel = weight_variable_accel([3, 3, 64, 128])
b_conv3_accel = bias_variable_accel([128])
h_conv3_accel = tf.nn.relu(conv2d_accel(h_conv2_accel_pool, W_conv3_accel, 1) + b_conv3_accel)

#fourth convolutional layer
W_conv4_accel = weight_variable_accel([3, 3, 128, 128])
b_conv4_accel = bias_variable_accel([128])
h_conv4_accel = tf.nn.relu(conv2d_accel(h_conv3_accel, W_conv4_accel, 1) + b_conv4_accel)

h_conv4_accel_pool = tf.nn.max_pool(h_conv4_accel, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')


# Block 3
#fifth convolutional layer
W_conv5_accel = weight_variable_accel([3, 3, 128, 256])
b_conv5_accel = bias_variable_accel([256])
h_conv5_accel = tf.nn.relu(conv2d_accel(h_conv4_accel_pool, W_conv5_accel, 1) + b_conv5_accel)

#sixth convolutional layer
W_conv6_accel = weight_variable_accel([3, 3, 256, 256])
b_conv6_accel = bias_variable_accel([256])
h_conv6_accel = tf.nn.relu(conv2d_accel(h_conv5_accel, W_conv6_accel, 1) + b_conv6_accel)

#seventh convolutional layer
W_conv7_accel = weight_variable_accel([3, 3, 256, 256])
b_conv7_accel = bias_variable_accel([256])
h_conv7_accel = tf.nn.relu(conv2d_accel(h_conv6_accel, W_conv7_accel, 1) + b_conv7_accel)

#eigth convolutional layer
W_conv8_accel = weight_variable_accel([3, 3, 256, 256])
b_conv8_accel = bias_variable_accel([256])
h_conv8_accel = tf.nn.relu(conv2d_accel(h_conv7_accel, W_conv8_accel, 1) + b_conv8_accel)

h_conv8_accel_pool = tf.nn.max_pool(h_conv8_accel, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')



# Block 4
#9th convolutional layer
W_conv9_accel = weight_variable_accel([3, 3, 256, 512])
b_conv9_accel = bias_variable_accel([512])
h_conv9_accel = tf.nn.relu(conv2d_accel(h_conv8_accel_pool, W_conv9_accel, 1) + b_conv9_accel)

#10th convolutional layer
W_conv10_accel = weight_variable_accel([3, 3, 512, 512])
b_conv10_accel = bias_variable_accel([512])
h_conv10_accel = tf.nn.relu(conv2d_accel(h_conv9_accel, W_conv10_accel, 1) + b_conv10_accel)

#11th convolutional layer
W_conv11_accel = weight_variable_accel([3, 3, 512, 512])
b_conv11_accel = bias_variable_accel([512])
h_conv11_accel = tf.nn.relu(conv2d_accel(h_conv10_accel, W_conv11_accel, 1) + b_conv11_accel)

#12th convolutional layer
W_conv12_accel = weight_variable_accel([3, 3, 512, 512])
b_conv12_accel = bias_variable_accel([512])
h_conv12_accel = tf.nn.relu(conv2d_accel(h_conv11_accel, W_conv12_accel, 1) + b_conv12_accel)

h_conv12_accel_pool = tf.nn.max_pool(h_conv12_accel, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')




# Block 5
#13th convolutional layer
W_conv13_accel = weight_variable_accel([3, 3, 512, 512])
b_conv13_accel = bias_variable_accel([512])
h_conv13_accel = tf.nn.relu(conv2d_accel(h_conv12_accel_pool, W_conv13_accel, 1) + b_conv13_accel)

#14th convolutional layer
W_conv14_accel = weight_variable_accel([3, 3, 512, 512])
b_conv14_accel = bias_variable_accel([512])
h_conv14_accel = tf.nn.relu(conv2d_accel(h_conv13_accel, W_conv14_accel, 1) + b_conv14_accel)

#15th convolutional layer
W_conv15_accel = weight_variable_accel([3, 3, 512, 512])
b_conv15_accel = bias_variable_accel([512])
h_conv15_accel = tf.nn.relu(conv2d_accel(h_conv14_accel, W_conv15_accel, 1) + b_conv15_accel)

#16th convolutional layer
W_conv16_accel = weight_variable_accel([3, 3, 512, 512])
b_conv16_accel = bias_variable_accel([512])
h_conv16_accel = tf.nn.relu(conv2d_accel(h_conv15_accel, W_conv16_accel, 1) + b_conv16_accel)

h_conv16_accel_pool = tf.nn.max_pool(h_conv16_accel, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')



#FCL 1
W_fc1_accel = weight_variable_accel([16*512, 512])
b_fc1_accel = bias_variable_accel([512])

# print(h_conv4_accel_pool.shape)
h_conv16_flat_accel = tf.reshape(h_conv16_accel_pool, [-1, 16*512])
h_fc1_accel = tf.nn.relu(tf.matmul(h_conv16_flat_accel, W_fc1_accel) + b_fc1_accel)

keep_prob_accel = tf.placeholder(tf.float32)
h_fc1_drop_accel = tf.nn.dropout(h_fc1_accel, keep_prob_accel)

#FCL 2
W_fc2_accel = weight_variable_accel([512, 1])
b_fc2_accel = bias_variable_accel([1])


y_accel = tf.multiply(tf.atan(tf.matmul(h_fc1_drop_accel, W_fc2_accel) + b_fc2_accel), 2)


# y_accel = tf.matmul(h_fc1_drop_accel, W_fc2_accel) + b_fc2_accel

# h_fc2_accel = tf.nn.relu(tf.matmul(h_fc1_drop_accel, W_fc2_accel) + b_fc2_accel)

# h_fc2_drop_accel = tf.nn.dropout(h_fc2_accel, keep_prob_accel)

# #FCL 3
# W_fc3_accel = weight_variable_accel([100, 50])
# b_fc3_accel = bias_variable_accel([50])

# h_fc3_accel = tf.nn.relu(tf.matmul(h_fc2_drop_accel, W_fc3_accel) + b_fc3_accel)

# h_fc3_drop_accel = tf.nn.dropout(h_fc3_accel, keep_prob_accel)

# #FCL 3
# W_fc4_accel = weight_variable_accel([50, 10])
# b_fc4_accel = bias_variable_accel([10])

# h_fc4_accel = tf.nn.relu(tf.matmul(h_fc3_drop_accel, W_fc4_accel) + b_fc4_accel)

# h_fc4_drop_accel = tf.nn.dropout(h_fc4_accel, keep_prob_accel)

# #Output
# W_fc5_accel = weight_variable_accel([10, 1])
# b_fc5_accel = bias_variable_accel([1])

# # if (tf.cond(tf.eisTheta = True)):

# # if (tf.cond(tf.equal(isTheta, tf.constant(True)), lambda: True, lambda: False)):
# # if (isTheta is not None):
# y_accel = tf.multiply(tf.atan(tf.matmul(h_fc4_drop_accel, W_fc5_accel) + b_fc5_accel), 2) #scale the atan output
# else:
# 	y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5 #linear output
