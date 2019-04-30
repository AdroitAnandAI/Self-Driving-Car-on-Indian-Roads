import numpy as np
import cv2
import scipy.misc
import math
from scipy import signal

# cap = cv2.VideoCapture('slow.flv')

#read data.txt
xs = []
ys = []
accels = []
with open("indian_dataset/data.txt") as f:
    for line in f:
        xs.append("indian_dataset/circuit2_x264.mp4 " + str(line.split()[0]).zfill(5) + ".jpg")
        ys.append(float(line.split()[1]))



out_imgpath = 'indian_dataset/optFlow.txt'
optF = open(out_imgpath, "w")

# Function to calculate distance 
def distance(x1 , y1 , x2 , y2): 
  
    # Calculating distance 
    return math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) * 1.0) 


# Take first frame and find corners in it
frame = scipy.misc.imread(xs[0], mode="RGB")
previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(previous_frame, mask = None, **feature_params)

frameNum = 0

while(frameNum < len(xs)):

    frame = scipy.misc.imread(xs[frameNum], mode="RGB")
    present_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(frame)
    p0 = cv2.goodFeaturesToTrack(present_frame, mask = None, **feature_params)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, present_frame, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    distSum = 0
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dist = distance (a, b, c, d)
        distSum += dist

    # When number of detected corners (good features) are less than 10 then distance becomes erroneous.
    if (good_old.shape[0] < 10):
        opticalParam = prev_opticalParam
    else:
        opticalParam = distSum/(math.pow(good_old.shape[0], 2)+math.pow(ys[frameNum], 2)+1)*100

    optF.write(str(opticalParam) + '\n')

    prev_opticalParam = opticalParam

    # Now update the previous frame and previous points
    previous_frame = present_frame.copy()

    frameNum += 1

cv2.destroyAllWindows()
optF.close()