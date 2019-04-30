import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import scipy
import cv2

# standard step - reset computation graphs
tf.reset_default_graph()

saveDirectory = './save'
L2NormConst = 0.001

import model_accel
all_vars = tf.trainable_variables()
# model_accel_vars = [k for k in all_vars if 'accel' in k.name]
print(model_accel.y_accel_.shape)
print(model_accel.y_accel.shape)
loss_accel = tf.reduce_mean(tf.square(tf.subtract(model_accel.y_accel_, model_accel.y_accel))) #+ tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * L2NormConst
train_step_accel = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6).minimize(loss_accel) # These are default parameters to the Adadelta optimizer
# train_step_accel = tf.train.AdamOptimizer().minimize(loss_accel)
sess_accel = tf.Session()
sess_accel.run(tf.global_variables_initializer())
saver_accel = tf.train.Saver(all_vars)


# create a summary to monitor cost tensor
tf.summary.scalar("loss_accel", loss_accel)

# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 50
batch_size = 80

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):

    xs, ys, accels, brakes = driving_data.LoadTrainBatch(batch_size, False)

    train_step_accel.run(feed_dict={model_accel.x_accel: xs, model_accel.y_accel_: accels, model_accel.keep_prob_accel: 0.5, model_accel.keep_prob_accel_conv: 0.25}, session = sess_accel)

    if i % 10 == 0:
      xs, ys, accels, brakes = driving_data.LoadValBatch(batch_size, False)
      loss_value_accel = loss_accel.eval(feed_dict={model_accel.x_accel:xs, model_accel.y_accel_: accels, model_accel.keep_prob_accel: 1.0, model_accel.keep_prob_accel_conv: 1.0}, session = sess_accel)
      print("Epoch: %d, Step: %d, Accel Loss: %g " % (epoch, epoch * batch_size + i, loss_value_accel))

    if i % batch_size == 0:
      if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)
      accel_checkpoint_path = os.path.join(saveDirectory, "model_accel.ckpt")
      filename_accel = saver_accel.save(sess_accel, accel_checkpoint_path)

  print("Model saved in file: %s" % filename_accel)


### To predict the output based on the above trained model###
xs = []
dataPath = "indian_dataset/"
fileNamePrefix = "circuit2_x264.mp4 "

with open(dataPath+"data.txt") as f:
    for line in f:
        xs.append(dataPath + fileNamePrefix + str(int(line.split()[0])).zfill(5)+".jpg")

i = 0
while(cv2.waitKey(10) != ord('q')):

    full_image = scipy.misc.imread(xs[i], mode="RGB")
    
    image = scipy.misc.imresize(full_image[-150:], [112, 112]) / 255.0
    # print(image.shape)
    # print(model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0}))
    # with g_accel.as_default():
    acceleration = model_accel.y_accel.eval(feed_dict={model_accel.x_accel: [image], model_accel.keep_prob_accel: 1.0, model_accel.keep_prob_accel_conv: 1.0}, session = sess_accel)[0][0]
    print(i,acceleration * 180.0 / scipy.pi)
    i += 1


# To Visualize CNN Layers to better interpretability. Base code obtained from:
# https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
# def getActivations(layer,stimuli, filename, steer, acceleration):
#     units = sess.run(layer,feed_dict={model.x:np.reshape(stimuli,[-1, 136, 240, 3],order='F'), model.y1_: np.reshape(steer, [-1, 1],order='F'), model.y2_: acceleration, model.keep_prob:1.0})
#     plotNNFilter(units, filename)


# def plotNNFilter(units, filename):
#     filters = units.shape[3]
#     plt.figure(1, figsize=(20,20))
#     n_columns = 6
#     n_rows = math.ceil(filters / n_columns) + 1
#     for i in range(filters):
#         plt.subplot(n_rows, n_columns, i+1)
#         plt.title('Filter ' + str(i))
#         plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
#     plt.savefig(filename)

# xs = []
# ys = []
# accels = []
# with open("indian_dataset/data.txt") as f:
#     for line in f:
#         xs.append("indian_dataset/circuit2_x264.mp4 " + str(line.split()[0]).zfill(5) + ".jpg")
#         ys.append(float(line.split()[1]) * scipy.pi / 180)
#         accels.append(float(line.split()[2]))

# frameNum = 4805
# full_image = scipy.misc.imread(xs[frameNum], mode="RGB")
# image = scipy.misc.imresize(full_image[-150:], [136, 240]) / 255.0
# cv2.imshow("Visualize CNN: input image", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
# cv2.imshow("Visualize CNN: input image", image)

# getActivations(model_steer.h_conv1,image, 'cnn-depict-conv1.jpg', ys[frameNum], accels[frameNum])
# getActivations(model_steer.h_conv2,image, 'cnn-depict-conv2.jpg', ys[frameNum], accels[frameNum])
# getActivations(model_steer.h_conv3,image, 'cnn-depict-conv3.jpg', ys[frameNum], accels[frameNum])
# getActivations(model_steer.h_conv4,image, 'cnn-depict-conv4.jpg', ys[frameNum], accels[frameNum])
# getActivations(model_steer.h_conv5,image, 'cnn-depict-conv5.jpg', ys[frameNum], accels[frameNum])
# getActivations(model_steer.h_conv6,image, 'cnn-depict-conv6.jpg', ys[frameNum], accels[frameNum])

sess_accel.close()