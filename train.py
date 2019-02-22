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

g_steer = tf.Graph() ## This is one graph
g_accel = tf.Graph() ## This is another graph
g_brake = tf.Graph() ## This is another graph

# train_vars = tf.trainable_variables()
# print(train_vars)

# with tf.name_scope("steer") as scope:
with g_steer.as_default():
    import model_steer
    # print(tf.trainable_variables())
    all_vars = tf.trainable_variables()
    # model_steer_vars = [k for k in all_vars if 'steer' in k.name]
    print(all_vars)
    loss_steer = tf.reduce_mean(tf.square(tf.subtract(model_steer.y_, model_steer.y))) + tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * L2NormConst
    train_step_steer = tf.train.AdamOptimizer(1e-4).minimize(loss_steer)
    sess_steer = tf.Session(graph = g_steer)
    sess_steer.run(tf.global_variables_initializer())
    saver_steer = tf.train.Saver(all_vars)

# with tf.name_scope("accel") as scope:
with g_accel.as_default():
    import model
    all_vars = tf.trainable_variables()
    # model_accel_vars = [k for k in all_vars if 'accel' in k.name]
    print(all_vars)
    loss_accel = tf.reduce_mean(tf.square(tf.subtract(model.y_accel_, model.y_accel))) + tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * L2NormConst
    train_step_accel = tf.train.AdamOptimizer(1e-4).minimize(loss_accel)
    sess_accel = tf.Session(graph = g_accel)
    sess_accel.run(tf.global_variables_initializer())
    saver_accel = tf.train.Saver(all_vars)

# with tf.name_scope("accel") as scope:
with g_brake.as_default():
    import model_brake
    all_vars = tf.trainable_variables()
    # model_accel_vars = [k for k in all_vars if 'accel' in k.name]
    print(all_vars)
    loss_brake = tf.reduce_mean(tf.square(tf.subtract(model_brake.y_brake_, model_brake.y_brake))) + tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * L2NormConst
    train_step_brake = tf.train.AdamOptimizer(1e-4).minimize(loss_brake)
    sess_brake = tf.Session(graph = g_brake)
    sess_brake.run(tf.global_variables_initializer())
    saver_brake = tf.train.Saver(all_vars)


# create a summary to monitor cost tensor
tf.summary.scalar("loss_accel", loss_accel)
tf.summary.scalar("loss_steer", loss_steer)
tf.summary.scalar("loss_brake", loss_brake)

# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# all_vars = tf.all_variables()
# model_steer_vars = [k for k in all_vars if 'steer' in k.name]
# model_accel_vars = [k for k in all_vars if 'accel' in k.name]
# print(all_vars)
# print(model_steer_vars)






# saver_steer = tf.train.Saver({v.op.name: v for v in model_steer_vars})
# saver_accel = tf.train.Saver({v.op.name: v for v in model_accel_vars})

# saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 0
batch_size = 200

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):

    xs, ys, accels, brakes = driving_data.LoadTrainBatch(batch_size)
    # with tf.name_scope("steer") as scope:
    train_step_steer.run(feed_dict={model_steer.x: xs, model_steer.y_: ys, model_steer.keep_prob: 0.8}, session = sess_steer)
    train_step_accel.run(feed_dict={model.x_accel: xs, model.y_accel_: accels, model.keep_prob_accel: 0.8}, session = sess_accel)
    train_step_brake.run(feed_dict={model_brake.x_brake: xs, model_brake.y_brake_: brakes, model_brake.keep_prob_brake: 0.8}, session = sess_brake)
    # xs, optFlow = driving_data.LoadCorrTrainBatch(batch_size)
    # train_step.run(feed_dict={model.x: xs, model.y_: optFlow, model.keep_prob: 0.8})
    if i % 10 == 0:
      # xs, optFlow  = driving_data.LoadCorrValBatch(batch_size)
      # loss_value = loss_accel.eval(feed_dict={model.x:xs, model.y_: optFlow, model.keep_prob: 1.0})
      xs, ys, accels, brakes = driving_data.LoadValBatch(batch_size)
      loss_value_steer = loss_steer.eval(feed_dict={model_steer.x:xs, model_steer.y_: ys, model_steer.keep_prob: 1.0}, session = sess_steer)
      loss_value_accel = loss_accel.eval(feed_dict={model.x_accel:xs, model.y_accel_: accels, model.keep_prob_accel: 1.0}, session = sess_accel)
      loss_value_brake = loss_brake.eval(feed_dict={model_brake.x_brake:xs, model_brake.y_brake_: brakes, model_brake.keep_prob_brake: 1.0}, session = sess_brake)
      # loss_value_accel = 0
      print("Epoch: %d, Step: %d, Steer Loss: %g Accel Loss: %g Brake Loss: %g" % (epoch, epoch * batch_size + i, loss_value_steer, loss_value_accel, loss_value_brake))

    # write logs at every iteration
    # summary = merged_summary_op.eval(feed_dict={model_steer.x:xs, model_steer.y_: ys, model_steer.keep_prob: 1.0, model.x:xs, model.y_: accels, model.keep_prob: 1.0})
    # summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: optFlow, model.keep_prob: 1.0})
    # summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)
      steer_checkpoint_path = os.path.join(saveDirectory, "model_steer.ckpt")
      accel_checkpoint_path = os.path.join(saveDirectory, "model_accel.ckpt")
      brake_checkpoint_path = os.path.join(saveDirectory, "model_brake.ckpt")
      filename_steer = saver_steer.save(sess_steer, steer_checkpoint_path)
      filename_accel = saver_accel.save(sess_accel, accel_checkpoint_path)
      filename_brake = saver_brake.save(sess_accel, brake_checkpoint_path)
      # filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename_steer)
  print("Model saved in file: %s" % filename_accel)
  print("Model saved in file: %s" % filename_brake)

sess_steer.close()
sess_accel.close()
sess_brake.close()



# To Visualize CNN Layers to better interpretability. Base code obtained from:
# https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
def getActivations(layer,stimuli, filename, steer, acceleration):
    units = sess.run(layer,feed_dict={model.x:np.reshape(stimuli,[-1, 136, 240, 3],order='F'), model.y1_: np.reshape(steer, [-1, 1],order='F'), model.y2_: acceleration, model.keep_prob:1.0})
    plotNNFilter(units, filename)


def plotNNFilter(units, filename):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.savefig(filename)

xs = []
ys = []
accels = []
with open("indian_dataset/data.txt") as f:
    for line in f:
        xs.append("indian_dataset/circuit2_x264.mp4 " + str(line.split()[0]).zfill(5) + ".jpg")
        ys.append(float(line.split()[1]) * scipy.pi / 180)
        accels.append(float(line.split()[2]))

frameNum = 4805
full_image = scipy.misc.imread(xs[frameNum], mode="RGB")
image = scipy.misc.imresize(full_image[-150:], [136, 240]) / 255.0
cv2.imshow("Visualize CNN: input image", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
# cv2.imshow("Visualize CNN: input image", image)

getActivations(model_steer.h_conv1,image, 'cnn-depict-conv1.jpg', ys[frameNum], accels[frameNum])
getActivations(model_steer.h_conv2,image, 'cnn-depict-conv2.jpg', ys[frameNum], accels[frameNum])
getActivations(model_steer.h_conv3,image, 'cnn-depict-conv3.jpg', ys[frameNum], accels[frameNum])
getActivations(model_steer.h_conv4,image, 'cnn-depict-conv4.jpg', ys[frameNum], accels[frameNum])
getActivations(model_steer.h_conv5,image, 'cnn-depict-conv5.jpg', ys[frameNum], accels[frameNum])
getActivations(model_steer.h_conv6,image, 'cnn-depict-conv6.jpg', ys[frameNum], accels[frameNum])
