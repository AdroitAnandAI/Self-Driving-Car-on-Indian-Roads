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

import model_brake
all_vars = tf.trainable_variables()
print(model_brake.y_brake_.shape)
print(model_brake.y_brake.shape)
loss_brake = tf.reduce_mean(tf.square(tf.subtract(model_brake.y_brake_, model_brake.y_brake))) #+ tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * L2NormConst
train_step_brake = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6).minimize(loss_brake) # These are default parameters to the Adadelta optimizer
sess_brake = tf.Session()
sess_brake.run(tf.global_variables_initializer())
saver_brake = tf.train.Saver(all_vars)

# create a summary to monitor cost tensor
tf.summary.scalar("loss_brake", loss_brake)

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

    train_step_brake.run(feed_dict={model_brake.x_brake: xs, model_brake.y_brake_: brakes, model_brake.keep_prob_brake: 0.5, model_brake.keep_prob_brake_conv: 0.25}, session = sess_brake)

    if i % 10 == 0:
      xs, ys, accels, brakes = driving_data.LoadValBatch(batch_size, False)
      loss_value_brake = loss_brake.eval(feed_dict={model_brake.x_brake:xs, model_brake.y_brake_: brakes, model_brake.keep_prob_brake: 1.0, model_brake.keep_prob_brake_conv: 1.0}, session = sess_brake)
      print("Epoch: %d, Step: %d, Accel Loss: %g " % (epoch, epoch * batch_size + i, loss_value_brake))

    if i % batch_size == 0:
      if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)
      brake_checkpoint_path = os.path.join(saveDirectory, "model_brake.ckpt")
      filename_brake = saver_brake.save(sess_brake, brake_checkpoint_path)

  print("Model saved in file: %s" % filename_brake)


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
    acceleration = model_brake.y_brake.eval(feed_dict={model_brake.x_brake: [image], model_brake.keep_prob_brake: 1.0, model_brake.keep_prob_brake_conv: 1.0}, session = sess_brake)[0][0]
    print(i,acceleration * 180.0 / scipy.pi)
    i += 1

sess_brake.close()