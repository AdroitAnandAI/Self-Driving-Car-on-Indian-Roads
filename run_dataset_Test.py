import tensorflow as tf
import scipy.misc
import cv2
from subprocess import call
import math
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# standard step - reset computation graphs
tf.reset_default_graph()


brake_path = 'brake_frames/brake'
gas_path = 'gas_frames/acceleration'
gear_path = 'gear_frames/gear'
steer_path = 'steer_frames/steering'

animate = False
gearShift = 0
max_gas_value = 5
max_brake_value = 5
alpha_accel = 0.1
beta_accel = 0.9
alpha_brake = 0.05
beta_brake = 0.95


# https://www.researchgate.net/figure/Hyperbolic-tangent-function-Plot-of-the-hyperbolic-tangent-function-y-tanhax-for_fig9_6266085
tanh_alpha = 0.18 # To decide the shape of the hyperbolic tangent function

g_steer = tf.Graph() ## This is one graph
g_accel = tf.Graph() ## This is another graph
g_brake = tf.Graph() ## This is another graph

with g_steer.as_default():
    import model
    all_vars = tf.trainable_variables()
    sess_steer = tf.Session(graph = g_steer)
    saver_steer = tf.train.Saver(all_vars)
    saver_steer.restore(sess_steer, "save/model_steer.ckpt")


with g_accel.as_default():
    import model_accel
    all_vars = tf.trainable_variables()
    sess_accel = tf.Session(graph = g_accel)
    saver_accel = tf.train.Saver(all_vars)
    saver_accel.restore(sess_accel, "save/model_accel.ckpt")


with g_brake.as_default():
    import model_brake
    all_vars = tf.trainable_variables()
    sess_brake = tf.Session(graph = g_brake)
    saver_brake = tf.train.Saver(all_vars)
    saver_brake.restore(sess_brake, "save/model_brake.ckpt")


def smoothFactor(predictedValue, previousValue):
    return 0.2 * pow(abs((predictedValue - previousValue)), 2.0 / 3.0) * (predictedValue - previousValue) / abs(predictedValue - previousValue + 1e-5)

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

dataPath = "Test_Track/"
fileNamePrefix = "TestTrack "


#read data.txt
xs = []
ys = []
accels = []
brakes = []
opticalFlow = []
gears = []
gearFeatures = []

with open(dataPath+"data.txt") as f:
    for line in f:
        xs.append(dataPath + fileNamePrefix + str(int(line.split()[1][:-4])).zfill(4)+".jpg")  

# # #get number of images
num_images = len(xs)

i = 0
gear = 0
smoothed_accel = 0
smoothed_brake = 0
predictedGears = []
considerPreviousGears = 10

print("Starting frameofvideo:" +str(i))
errorDegreeAccum = 0


while(cv2.waitKey(10) != ord('q') and i < num_images-1):

    full_image = scipy.misc.imread(xs[i], mode="RGB")

    image_steer = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image_steer], model.keep_prob: 1.0}, session = sess_steer)[0][0] * 180.0 / scipy.pi

    pred_deg = round(degrees, 2)
    print("Steering angle: " + str(degrees) + " (pred)\t")

    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += smoothFactor(degrees, smoothed_angle)
    i += 1

    # steer_frame_map = [1, 190, 369] 
    steer_frame = min(max(1, int(190 + 2*smoothed_angle)), 369)
    print(steer_path + " " + str(steer_frame).zfill(3)+".jpg")
    steer_img = cv2.imread(steer_path + " " + str(steer_frame).zfill(3)+".jpg")
    cv2.imshow("Steering Wheel", steer_img)

    if (animate):
        # To show the braking animation
        brake_img = cv2.imread(brake_path + " " + str(int(smoothed_brake*2+1)).zfill(2)+".jpg")
        brake_img_small = cv2.resize(brake_img, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow('Brake Pedal', brake_img_small)

        # To show the gal pedal animation
        accel_img = cv2.imread(gas_path + " " + str(int(smoothed_accel*2+1)).zfill(2)+".jpg")
        accel_img_small = cv2.resize(accel_img, (0,0), fx=0.52, fy=0.52) 
        cv2.imshow('Acceleration Pedal', accel_img_small)

        # To show the gear animation: Frames are mapped as below.
        # 0-10: 1st gear, 11-20: 2nd gear, 21-30: 3rd gear, 
        # 30-37: 4th gear, 37-45: 5th gear, 45-50: 5th to neutral.

        # Frames correspond to Neutral, 1st, 2nd, 3rd, 4th and 5th gears
        gear_frame_map = [1, 12, 22, 30, 37, 45] 
        # Frames correspond to neutral between gears
        neutral_frame_map = [16, 26, 33, 42, 50]

        if (gearShift != 0):

            # To animate shift from any gear to neutral
            if (gearShift == -2):
                init_frame = gear_frame_map [gear]
                end_frame = neutral_frame_map [gear-1]

                for frame in range(init_frame, end_frame, 1):
                    gear_img = cv2.imread(gear_path + " " + str(frame).zfill(2)+".jpg")
                    cv2.imshow('Gear-Transmission', gear_img)
                    cv2.waitKey(60)

                gear = 0

            else: # when gearShift = -1 and +1 (gear up or down)
                print("gear = " + str(gear) + " Gear Shift = " + str(gearShift))

                init_frame = gear_frame_map [gear + gearShift*-1]
                end_frame = gear_frame_map [gear]
                print("init_frame = " + str(init_frame) + " end_frame = " + str(end_frame))

                for frame in range(init_frame, end_frame, gearShift):
                    # File read should confirm with the video to jpg frame extractor software
                    gear_img = cv2.imread(gear_path + " " + str(frame).zfill(2)+".jpg")
                    cv2.imshow('Gear-Transmission', gear_img)
                    cv2.waitKey(60)

            gearShift = 0

print("Accumulated Steering Angle Error: " + str(errorDegreeAccum))

cv2.destroyAllWindows()
