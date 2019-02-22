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


out_imgpath = 'indian_dataset/corr/optFlow.txt'
optF = open(out_imgpath, "w")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


# Function to calculate distance 
def distance(x1 , y1 , x2 , y2): 
  
    # Calculating distance 
    return math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) * 1.0) 


# Take first frame and find corners in it
frame = scipy.misc.imread(xs[0], mode="RGB")
previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(previous_frame, mask = None, **feature_params)


# height = frame.shape[0]
# width = frame.shape[1]

# Create a mask image for drawing purposes
# mask = np.zeros_like(frame)

# featureMask = np.zeros(frame.shape, dtype=np.uint8)

frameNum = 0

while(frameNum < len(xs)):

    frame = scipy.misc.imread(xs[frameNum], mode="RGB")
    present_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # not to make the mask messy
    # if (frameNum % 100 == 0):
        # previous_frame = present_frame
    mask = np.zeros_like(frame)
    p0 = cv2.goodFeaturesToTrack(present_frame, mask = None, **feature_params)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, present_frame, p0, None, **lk_params)


    # flow = cv2.calcOpticalFlowFarneback(previous_frame, present_frame, flow=None,pyr_scale=0.5, levels=1, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
    # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])


    # if the all the optical flow objects go out of the frame then p1 will be none
    # if p1 is None:
    #    print("features changed!")
    #    p0 = cv2.goodFeaturesToTrack(present_frame, mask = None, **feature_params)
    #    p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, present_frame, p0, None, **lk_params)

    # print(p1)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # print("shape old = " + str(good_old.shape))
    # print("shape new = " + str(good_new.shape))

    distSum = 0
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dist = distance (a, b, c, d)
        distSum += dist

        # print("a = " + str(a) + "b = " + str(b) + "\t Distance = " + str(dist))

    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     # print(mask.shape)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)

    # print(ys[frameNum])

    # When number of detected corners (good features) are less than 10 then distance becomes erroneous.
    if (good_old.shape[0] < 10):
        opticalParam = prev_opticalParam
    else:
        opticalParam = distSum/(math.pow(good_old.shape[0], 2)+math.pow(ys[frameNum], 2)+1)*100

    # print(math.abs(a - frame.shape[1]/2))

    # opticalParam = distSum/(good_old.shape[0]+math.pow(ys[frameNum], 2)+1e-5)*100

    # print("%d %.6f \n"%(good_old.shape[0], opticalParam))
    optF.write(str(opticalParam) + '\n')

    prev_opticalParam = opticalParam
    # print(present_frame.shape)
    
    # print("you are here")
    # present_frame_small = scipy.misc.imresize(present_frame[-150:], [66, 200]) / 255.0
    # previous_frame_small = scipy.misc.imresize(previous_frame[-150:], [66, 200]) / 255.0

    # corrimage = signal.correlate2d (present_frame_small, previous_frame_small)

    # scipy.misc.imsave("indian_dataset/corr/circuit2_x264.mp4 "+str(frameNum).zfill(5)+".png", corrimage)

    # cv2.imshow('frame',img)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break

    # Now update the previous frame and previous points
    previous_frame = present_frame.copy()
    # p0 = good_new.reshape(-1,1,2)


    frameNum += 1
    # previous_frame = 
    # print(frameNum)

cv2.destroyAllWindows()
# cap.release()

optF.close()