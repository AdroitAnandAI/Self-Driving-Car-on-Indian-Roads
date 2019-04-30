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


### To predict the output based on the above trained model### - THE BUG IS IN THE CODE ABOVE!
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

sess_accel.close()