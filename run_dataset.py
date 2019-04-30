import tensorflow as tf
import scipy.misc
# import model
# import model_steer
# import model_nonTheta
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

animate = True
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

# dataPath = "indian_dataset/"
dataPath = "Test_Track/"
corrDataPath = "indian_dataset/corr/"
# fileNamePrefix = "circuit2_x264.mp4 "
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
        xs.append(dataPath + fileNamePrefix + str(int(line.split()[0])).zfill(5)+".jpg")  
        # No need to convert to radians as here we dont use for training.
        steer_value = float(line.split()[1]) 
        accel_value = float(line.split()[2])
        brake_value = float(line.split()[3])
        gear_value = float(line.split()[4])
        ys.append(steer_value)
        accels.append(accel_value)
        brakes.append(brake_value)
        gears.append(gear_value)
        gearFeatures.append([steer_value, accel_value, brake_value])

i = 0
with open(corrDataPath+"optFlow.txt") as f:
    for line in f:
        opticalFlow.append(float(line.split()[0]))
        gearFeatures[i].append(float(line.split()[0]))
        i += 1

gearModel = RandomForestClassifier()
gearModel.fit(np.array(gearFeatures), np.array(gears))

# #get number of images
num_images = len(ys)

i = 0
gear = 0
smoothed_accel = 0
smoothed_brake = 0
predictedGears = []
considerPreviousGears = 10

print("Starting frameofvideo:" +str(i))
errorDegreeAccum = 0

# To display neutral gear
gear_img = cv2.imread(gear_path + " " + str(1).zfill(2)+".jpg")
cv2.imshow('Gear-Transmission', gear_img)


while(cv2.waitKey(10) != ord('q') and i < num_images-1):

    full_image = scipy.misc.imread(xs[i], mode="RGB")
    
    image = scipy.misc.imresize(full_image[-150:], [112, 112]) / 255.0
    image_steer = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    acceleration = model_accel.y_accel.eval(feed_dict={model_accel.x_accel: [image], model_accel.keep_prob_accel: 1.0, model_accel.keep_prob_accel_conv: 1.0}, session = sess_accel)[0][0] * 180.0 / scipy.pi
    degrees = model.y.eval(feed_dict={model.x: [image_steer], model.keep_prob: 1.0}, session = sess_steer)[0][0] * 180.0 / scipy.pi
    brake = model_brake.y_brake.eval(feed_dict={model_brake.x_brake: [image], model_brake.keep_prob_brake: 1.0, model_brake.keep_prob_brake_conv: 1.0}, session = sess_brake)[0][0] * 180.0 / scipy.pi

    # To squash the value between 0 and max_gas_value
    optical_adjusted_gas = max_gas_value*math.tanh(tanh_alpha*opticalFlow[i])
    optical_adjusted_brake = max_brake_value*math.tanh(tanh_alpha*opticalFlow[i])

    accel_integrated = optical_adjusted_gas*alpha_accel + accels[i]*beta_accel
    brake_integrated = optical_adjusted_brake*alpha_brake + brakes[i]*beta_brake

    print("\nAcceleration: " + str(acceleration) + " (pred)\t\t" + str(round(accels[i], 2)))
    print("Brake: " + str(brake_integrated) + " (pred)\t\t" + str(round(brakes[i], 2)))

    smoothed_accel += smoothFactor(accel_integrated, smoothed_accel)
    smoothed_brake += smoothFactor(brake_integrated, smoothed_brake)

    predictedGear = gearModel.predict(np.array(gearFeatures[i]).reshape(1, -1))
    print("Gear = " + str(predictedGear[0]))
    predictedGears.append(predictedGear)

    # lazy check to see whether all gear predictions in previous 'x' frames same as current prediction
    # if same then take the gear value seriously.
    previousGears = predictedGears[-considerPreviousGears:]
    if (sum(previousGears)/len(previousGears) == predictedGear):
        takeGearSeriously = True
    else:
        takeGearSeriously = False

    # if repeated frames predict a different gear then change the gear.
    if (predictedGear[0] != gear and abs(gear - predictedGear[0]) == 1 and takeGearSeriously): # if gear shift
        gearShift = int(predictedGear[0] - gear)
        gear = int(predictedGear[0])
        print("GEAR CHANGED!!!")

    pred_deg = round(degrees, 2)
    actual_deg = round(ys[i], 2)
    diff_deg = round(abs(pred_deg - actual_deg), 2)
    errorDegreeAccum += diff_deg

    print("Steering angle: " + str(degrees) + " (pred)\t" + str(actual_deg) + " (actual)\t\t" + "Angular Error: " + str(diff_deg))

    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += smoothFactor(degrees, smoothed_angle)
    i += 1

    # Frames correspond to frame numbers of turn end points in steer video.
    # Degrees in Frame: -180 to +180 
    steer_frame_map = [1, 190, 369] 
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
                    # gear_img_small = cv2.resize(gear_img, (0,0), fx=0.5, fy=0.5) 
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
