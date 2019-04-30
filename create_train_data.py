# This program intends to make the car console interface to generate car driving training data.

import cv2
import time

brake_path = 'brake_frames/brake'
gas_path = 'gas_frames/acceleration'
gear_path = 'gear_frames/gear'

out_imgpath = 'indian_dataset/data.txt'
f = open(out_imgpath, "w")

smoothed_angle = 0
degrees = 1
gas_pedal = 0
brake_pedal = 0
gear = 0
gearShift = 0
frame = 1
writeTime = -1
frameRate = []
keepUpArrowPressed = False

# in training mode, animation should be switched off for syncing.
animate = True

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

# To display neutral gear
gear_img = cv2.imread(gear_path + " " + str(1).zfill(2)+".jpg")
cv2.imshow('Gear-Transmission', gear_img)


start = int(round(time.time() * 1000))

while(True):

	# to return full key code so that arrow keys also can be detected.
	k = cv2.waitKeyEx(1)

	# Scan Codes for arrow keys
	# 2490368 up - for gas_pedal
	# 2621440 down - for break
	# 2424832 left - to turn left
	# 2555904 right - to turn right
	# a = to shift gear up
	# z = to shift gear down
	# n = to shift to neutral (from all gears)

	if (k==2490368):
		print('up')
		if (gas_pedal < 5): # max value of gas = 5
			gas_pedal += 0.5
			keepUpArrowPressed = True
		print('gas = ' + str(gas_pedal))
	elif (k==2621440):
		print('down')
		if (brake_pedal < 5): # max value of break = 5
			brake_pedal += 0.5
			keepUpArrowPressed = False
		print('break = ' + str(brake_pedal))
	elif (k==2555904):
		print('right')
		degrees += 3
		print('Angle = ' + str(degrees))
	elif (k==2424832):
		print('left')
		degrees -= 3
		print('Angle = ' + str(degrees))
	elif (k == ord('a')):
		print("Gear up")
		if (gear < 5):
			gear += 1
			gearShift = 1
		print('Gear = ' + str(gear))
	elif (k == ord('z')):
		print("Gear down")
		if (gear > 0):
			gear -= 1
			gearShift = -1
		print('Gear = ' + str(gear))
	elif (k == ord('g')):
		print("Neutral Gear")
		if (gear != 0):
			gearShift = -2
		print('Gear = ' + str(gear))
	else:
		if (gas_pedal > 0.01 and  not keepUpArrowPressed): # min value of gas = 0
			gas_pedal -= 0.005
		if (brake_pedal > 0.01): # min value of break = 0
			brake_pedal -= 0.005

		gearShift = 0

	
	smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
	M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imshow("steering wheel", dst)

	if (animate):
		# To show the braking animation
		brake_img = cv2.imread(brake_path + " " + str(int(brake_pedal*2+1)).zfill(2)+".jpg")
		brake_img_small = cv2.resize(brake_img, (0,0), fx=0.5, fy=0.5) 
		cv2.imshow('Brake Pedal', brake_img_small)

		# To show the gal pedal animation
		accel_img = cv2.imread(gas_path + " " + str(int(gas_pedal*2+1)).zfill(2)+".jpg")
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

				for i in range(init_frame, end_frame, 1):
					gear_img = cv2.imread(gear_path + " " + str(i).zfill(2)+".jpg")
					cv2.imshow('Gear-Transmission', gear_img)
					cv2.waitKey(1)

				gear = 0

			else: # when gearShift = -1 and +1 (gear up or down)
				init_frame = gear_frame_map [gear + gearShift*-1]
				end_frame = gear_frame_map [gear]
				print("init_frame = " + str(init_frame) + " end_frame = " + str(end_frame))

				for i in range(init_frame, end_frame, gearShift):
					# File read should confirm with the video to jpg frame extractor software
					gear_img = cv2.imread(gear_path + " " + str(i).zfill(2)+".jpg")
					# gear_img_small = cv2.resize(gear_img, (0,0), fx=0.5, fy=0.5) 
					cv2.imshow('Gear-Transmission', gear_img)
					cv2.waitKey(1)


	print('gas = ' + str(gas_pedal))
	print('break = ' + str(brake_pedal))

	# 1 driving second = 30 frames (extracted). We need to sync with this frame rate to find corresponding params of each frame.
	# Hence we will write the parameter values for acceleration, brake, gear and steering every 33 milliseconds. To average out 
	# around 33 ms, we need to tune the write frequency cut off, considering the running time of each while loop.
	now = int(round(time.time() * 1000000))
	
	if (now - writeTime >= 30001):
		print("time = " + str(now - writeTime))
		if (now - writeTime < 100000):
			frameRate.append(now - writeTime)

		# print(gas_pedal)
		print('gas = ' + str(gas_pedal))
		print('break = ' + str(brake_pedal))
		f.write(str(frame) + " " + str(degrees) + " " + str(round(gas_pedal, 2)) + " " + str(round(brake_pedal, 2)) + " " + str(gear) + '\n')
		writeTime = int(round(time.time() * 1000000))
		# to denote the frame of the input video
		frame += 1

	# To exit interface, press 'q'
	if (k == ord('q')):
		break


end = int(round(time.time() * 1000))

print("\n\nIdeal average write rate should be 33.33")
print("Average Write Rate = " + str(round(sum(frameRate)/ (1000*len(frameRate)), 2)))
print("Total Running Time = " + str(float((end-start)/1000)) + " seconds." + "Number of Frames in data.txt should be " + str(int((end-start)*30/1000)))

f.close()