import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, validation_curve, KFold, cross_val_score, GridSearchCV
from svm import SVM
import warnings
warnings.filterwarnings('ignore')

def predict(clf,X):
    return np.dot(clf[:-1],X) + clf[-1]

training_dataframe = pd.read_csv('Input/train.csv')

testing_dataframe = pd.read_csv('Input/test.csv')

rcount = int(0.2*training_dataframe.shape[0])
subset_training_dataframe = training_dataframe.sample(n=rcount)
subset_training_dataframe_train, subset_training_dataframe_test = train_test_split(subset_training_dataframe, test_size = 0.5, random_state = 4)
# print(np.shape(subset_training_dataframe_train))
# print(np.shape(subset_training_dataframe_test))
classifiers = []

#--------OVR--------------
# for i in range (0,10):

# 	X_i = subset_training_dataframe_train[subset_training_dataframe_train['label']==i].drop("label", axis = 1)
# 	y_i = subset_training_dataframe_train[subset_training_dataframe_train['label']==i].label
# 	rcount = int(0.15*subset_training_dataframe_train.shape[0])
# 	_subset_training_dataframe_train = subset_training_dataframe_train.sample(n=rcount)
# 	X_j = _subset_training_dataframe_train[subset_training_dataframe_train['label']!=i].drop("label", axis = 1)
# 	y_j = _subset_training_dataframe_train[subset_training_dataframe_train['label']!=i].label


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

# 	classifiers.append(np.append(model.w,[-model.b]))

# 	print(np.shape(classifiers))

# with open('OVRmodel.txt','wb') as f:
# 	np.savetxt(f, classifiers)


# ##---------------Predict_Func---------------------

# classifiers = np.loadtxt("OVRmodel.txt")
# def predict_class(X, classifiers):
#     predictions = np.zeros(len(classifiers))
#     for idx, clf in enumerate(classifiers):
#         predictions[idx] = predict(clf,X)
#         #print(predictions[idx])
#     out = np.argmin(predictions[:])
#     return out


#-------------------------------------------


#------------------OVO----------------------
for i in range (0,9):
	X_i = subset_training_dataframe_train[subset_training_dataframe_train['label']==i].drop("label", axis = 1)
	y_i = subset_training_dataframe_train[subset_training_dataframe_train['label']==i].label
	for j in range (i+1,10):
		X_j = subset_training_dataframe_train[subset_training_dataframe_train['label']==j].drop("label", axis = 1)
		y_j = subset_training_dataframe_train[subset_training_dataframe_train['label']==j].label


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

		model = SVM()

		model.fit(Xtemp,ytemp)

	#	print(calc_acc(y_test,y_predict))

		classifiers.append(np.append(model.w,[-model.b]))

	#print(np.shape(classifiers))



with open('OVOmodel.txt','wb') as f:
	np.savetxt(f, classifiers)

#---------------Predict_Func--------------------------
classifiers = np.loadtxt("OVOmodel.txt")
def predict_class(X, classifiers):
    predictions = np.zeros(10)
    pre=int(0)
    pos=int(1)
    for idx, clf in enumerate(classifiers):
        if(pos==10):
        	pre = pre+1
        	pos = pre+1
        _predict = predict(clf,X)
        if(_predict>0):
        	predictions[pos]=predictions[pos]+1
        	#predictions[pos]=max(predictions[pos],abs(_predict))
        	#predictions[pos]=predictions[pos]+abs(_predict)
        else:
        	predictions[pre]=predictions[pre]+1
        	#predictions[pre]=max(predictions[pre],abs(_predict))
        	#predictions[pre]=predictions[pre]+abs(_predict)
        pos = pos + 1

    out = np.argmax(predictions[:])
    return out

#--------------------------------------------------
#------------------PREDICT_PERFORMANCE-------------
X_test = subset_training_dataframe_test.drop("label", axis = 1)
y_test = subset_training_dataframe_test.label.values.astype(int)
X_test = scale(X_test)
sum = 0
print('test:')
for i in range(0,X_test.shape[0]):
	y_predict = predict_class(X_test[i,:],classifiers)
	# print("predict:")
	# print(y_predict)
	# print("real:")
	# print(y_test[i])
	if(y_predict==y_test[i]):
		sum = sum + 1
print(sum/y_test.shape[0])
print('')
#--------------------------------------------------
X_test = subset_training_dataframe_train.drop("label", axis = 1)
y_test = subset_training_dataframe_train.label.values.astype(int)
X_test = scale(X_test)
sum = 0
print('train:')
for i in range(0,X_test.shape[0]):
	y_predict = predict_class(X_test[i,:],classifiers)
	if(y_predict==y_test[i]):
		sum = sum + 1
print(sum/y_test.shape[0])

