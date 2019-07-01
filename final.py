import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#print(np.shape(classifiers))

def predict(clf,X):
    return np.dot(clf[:-1],X) + clf[-1]
#------------OVR------------------------------
# classifiers = np.loadtxt("OVRmodel.txt")
# def predict_class(X, classifiers):
#     predictions = np.zeros(len(classifiers))
#     for idx, clf in enumerate(classifiers):
#         predictions[idx] = predict(clf,X)
#         #print(predictions[idx])
#     out = np.argmin(predictions[:])
#     return out
#----------------------------------------------
#------------OVO------------------------------
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
#----------------------------------------------

# training_dataframe = pd.read_csv('Input/train.csv')
# rcount = int(0.5*training_dataframe.shape[0])
# subset_training_dataframe = training_dataframe.sample(n=rcount)
# X = subset_training_dataframe.drop("label", axis = 1)
# y = subset_training_dataframe.label.values.astype(int)
# X = scale(X)
# sum = 0
# for i in range(0,X.shape[0]):
# 	y_predict = predict_class(X[i,:],classifiers)
# 	print("predict:")
# 	print(y_predict)
# 	print("real:")
# 	print(y[i])
# 	if(y_predict==y[i]):
# 		sum = sum + 1
# print(sum/y.shape[0])





#--------------Predict_TEST.csv----------
testing_dataframe = pd.read_csv('Input/test.csv')
rcount = int(0.005*testing_dataframe.shape[0])
testing_dataframe = testing_dataframe.sample(n=rcount)
testing_dataframe = scale(testing_dataframe)
print(testing_dataframe[1])
for i in range(0,10):
	_2d = testing_dataframe[i].reshape(28,28)
	plt.title(f'Predicted Label: {predict_class(testing_dataframe[i],classifiers)}')
	plt.imshow(_2d)
	plt.show()

#-------------DRAWimg-----------------
# img = cv2.imread('28x28(1).png',0)
# imgplt = plt.imshow(img,cmap='gray')
# plt.show()

# imgg =[]

# for i in img:
# 	for j in i:
#   		imgg.append(j)
# imgg = np.array(imgg)
# imgg = scale(imgg)

# print(np.shape(imgg))
# print(predict_class(imgg,classifiers))
#------------------------------------------


