import scipy.misc
import random
import numpy as np

xs = []
ys = []
accels = []
brake = []
# gear = []
opticalFlow = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
corr_train_batch_pointer = 0
corr_val_batch_pointer = 0

dataPath = "indian_dataset/"
corrDataPath = "indian_dataset/corr/"
fileNamePrefix = "circuit2_x264.mp4 "
#read data.txt
with open(dataPath+"data.txt") as f:
# with open("driving_dataset/data.txt") as f:
    for line in f:
        # xs.append("driving_dataset/" + line.split()[0])
        xs.append(dataPath + fileNamePrefix + str(int(line.split()[0])).zfill(5)+".jpg")

        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)
        accels.append(float(line.split()[2]) * scipy.pi / 180)
        brake.append(float(line.split()[3]) *scipy.pi / 180)
        # gear.append(float(line.split()[4]) *scipy.pi / 180)


with open(corrDataPath+"optFlow.txt") as f:
# with open("driving_dataset/data.txt") as f:
    for line in f:
        # xs.append("driving_dataset/" + line.split()[0])
        opticalFlow.append(float(line.split()[0]) * scipy.pi / 180)


#get number of images
num_images = len(xs)

# #shuffle list of images
# c = list(zip(xs, ys))
# random.shuffle(c)
# xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]
train_accels = accels[:int(len(xs) * 0.8)]
train_brake = brake[:int(len(xs) * 0.8)]
train_optFlow = opticalFlow[:int(len(opticalFlow) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]
val_accels = accels[-int(len(xs) * 0.2):]
val_brake = brake[-int(len(xs) * 0.2):]
val_optFlow = opticalFlow[-int(len(opticalFlow) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    accels_out = []
    brake_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
        accels_out.append([train_accels[(train_batch_pointer + i) % num_train_images]])
        brake_out.append([train_brake[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out, accels_out, brake_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    accels_out = []
    brake_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
        accels_out.append([val_accels[(val_batch_pointer + i) % num_val_images]])
        brake_out.append([val_brake[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out, accels_out, brake_out


##############################################################################################

def LoadCorrTrainBatch(batch_size):
    global corr_train_batch_pointer
    x_out = []
    # y_out = []
    optFlow_out = []
    for i in range(0, batch_size):

        present_frame = (corr_train_batch_pointer + i) % num_train_images

        img = scipy.misc.imresize(scipy.misc.imread(corrDataPath + fileNamePrefix + train_xs[present_frame]), [66, 200]) / 255.0
        stacked_img = np.stack((img,)*3, axis=-1)

        x_out.append(stacked_img)
        # print(opticalFlow[present_frame])
        optFlow_out.append([opticalFlow[present_frame]])

        # present_frame = (corr_train_batch_pointer + i) % num_train_images
        # previous_frame = present_frame - 1
        # present_image = scipy.misc.imresize(scipy.misc.imread(train_xs[present_frame])[-150:], [66, 200]).sum(-1) / 255.0
        # previous_image = scipy.misc.imresize(scipy.misc.imread(train_xs[previous_frame])[-150:], [66, 200]).sum(-1) / 255.0
        # print(present_image.shape)

        # x_out.append(signal.correlate2d (present_image, previous_image))
        # y_out.append([train_ys[present_frame]])
        # accels_out.append([accels[present_frame]])
        # opticalFlow[present_frame]

    corr_train_batch_pointer += batch_size
    return x_out, optFlow_out

def LoadCorrValBatch(batch_size):
    global corr_val_batch_pointer
    x_out = []
    # y_out = []
    # accels_out = []
    optFlow_out = []
    for i in range(0, batch_size):

        present_frame = (corr_val_batch_pointer + i) % num_val_images

        img = scipy.misc.imresize(scipy.misc.imread(corrDataPath + fileNamePrefix + val_xs[present_frame]), [66, 200]) / 255.0
        stacked_img = np.stack((img,)*3, axis=-1)

        x_out.append(stacked_img)
        optFlow_out.append([opticalFlow[present_frame]])
        
        # previous_frame = present_frame - 1
        # present_image = scipy.misc.imresize(scipy.misc.imread(train_xs[present_frame])[-150:], [66, 200]).sum(-1) / 255.0
        # previous_image = scipy.misc.imresize(scipy.misc.imread(train_xs[previous_frame])[-150:], [66, 200]).sum(-1) / 255.0
        # x_out.append(signal.correlate2d (present_image, previous_image))
        # y_out.append([val_ys[present_frame]])
        # accels_out.append([accels[present_frame]])

    corr_val_batch_pointer += batch_size
    return x_out, optFlow_out