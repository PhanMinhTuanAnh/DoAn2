import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, validation_curve, KFold, cross_val_score, GridSearchCV
from svm import SVM
import warnings
warnings.filterwarnings('ignore')

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)


training_dataframe = pd.read_csv('Input/train.csv')

testing_dataframe = pd.read_csv('Input/test.csv')

rcount = int(0.2*training_dataframe.shape[0])
subset_training_dataframe = training_dataframe.sample(n=rcount)

classifiers = []

#--------OVR--------------
# for i in range (0,10):

# 	X_i = subset_training_dataframe[subset_training_dataframe['label']==i].drop("label", axis = 1)
# 	y_i = subset_training_dataframe[subset_training_dataframe['label']==i].label
# 	rcount = int(0.15*subset_training_dataframe.shape[0])
# 	_subset_training_dataframe = subset_training_dataframe.sample(n=rcount)
# 	X_j = _subset_training_dataframe[subset_training_dataframe['label']!=i].drop("label", axis = 1)
# 	y_j = _subset_training_dataframe[subset_training_dataframe['label']!=i].label


# 	Xtemp = X_i.append(X_j)
# 	ytemp = y_i.append(y_j)
# 	ytemp = ytemp.values.astype(int)
# 	print(ytemp)
# 	for z in range(0,ytemp.shape[0]):
# 		if(ytemp[z]==i):
# 			ytemp[z]=-1
# 	for z in range(0,ytemp.shape[0]):
# 		if(ytemp[z]!=-1):
# 			ytemp[z]=1
# 	print(ytemp)
# 	Xtemp = scale(Xtemp)


# 	X_train, X_test, y_train, y_test = train_test_split(Xtemp,ytemp, test_size = 0.2)

# 	model = SVM()

# 	model.fit(X_train,y_train)

# 	y_predict = model.predict(X_test)

# #	print(calc_acc(y_test,y_predict))

# 	classifiers.append(np.append(model.w,[-model.b]))

# 	print(np.shape(classifiers))

# with open('OVRmodel.txt','wb') as f:
# 	np.savetxt(f, classifiers)
#-------------------------------------------


#------------------OVO----------------------
for i in range (0,9):
	X_i = subset_training_dataframe[subset_training_dataframe['label']==i].drop("label", axis = 1)
	y_i = subset_training_dataframe[subset_training_dataframe['label']==i].label
	for j in range (i+1,10):
		X_j = subset_training_dataframe[subset_training_dataframe['label']==j].drop("label", axis = 1)
		y_j = subset_training_dataframe[subset_training_dataframe['label']==j].label


		Xtemp = X_i.append(X_j)
		ytemp = y_i.append(y_j)
		ytemp = ytemp.values.astype(int)
		#print(ytemp)
		for z in range(0,ytemp.shape[0]):
			if(ytemp[z]==i):
				ytemp[z]=-1
			else:
				ytemp[z]=1
		#print(ytemp)
		Xtemp = scale(Xtemp)

		X_train, X_test, y_train, y_test = train_test_split(Xtemp,ytemp, test_size = 0.2)

		model = SVM()

		model.fit(X_train,y_train)

		y_predict = model.predict(X_test)

		print(calc_acc(y_test,y_predict))

		classifiers.append(np.append(model.w,[-model.b]))

	#print(np.shape(classifiers))



with open('OVOmodel.txt','wb') as f:
	np.savetxt(f, classifiers)
#--------------------------------------------------



# img = cv2.imread('28x28.png',0)
# imgplt = plt.imshow(img,cmap='gray')
# plt.show()

# imgg =[]

# for i in img:
# 	for j in i:
#  		imgg.append(j)
# imgg = np.array(imgg)
# imgg = scale(imgg)

# y_predict = model.predict(imgg)
# print(y_predict)