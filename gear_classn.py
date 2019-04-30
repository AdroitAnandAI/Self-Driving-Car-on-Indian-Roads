from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import scipy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


xs = []
ys = []
accels = []
brake = []
opticalFlow = []
gear = []
gearFeatures = []

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
        steer_value = float(line.split()[1]) 
        accel_value = float(line.split()[2])
        brake_value = float(line.split()[3])
        gear_value = float(line.split()[4])
        ys.append(steer_value)
        accels.append(accel_value)
        brake.append(brake_value)
        gear.append(gear_value)
        gearFeatures.append([steer_value, accel_value, brake_value])
		# print(float(line.split()[1]) * scipy.pi / 180)



i = 0
with open(corrDataPath+"optFlow.txt") as f:
# with open("driving_dataset/data.txt") as f:
    for line in f:
        # xs.append("driving_dataset/" + line.split()[0])
        opticalFlow.append(float(line.split()[0]) * scipy.pi / 180)
        gearFeatures[i].append(float(line.split()[0]))
        i += 1

seed = 1

X_train, X_test, y_train, y_test = train_test_split(np.array(gearFeatures), np.array(gear), test_size=0.33, random_state=seed)

model = RandomForestClassifier()
model.fit(X_train, y_train)

#Make predictions for test data
y_pred = model.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))