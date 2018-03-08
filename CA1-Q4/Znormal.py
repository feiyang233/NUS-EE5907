import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn import preprocessing

mat = io.loadmat('spamData.mat')# <class 'dict'>
Xtrain=mat['Xtrain'] #feature  (3065, 57)
ytrain=mat['ytrain'] #<class 'numpy.ndarray'> (3065, 1)

Xtest=mat['Xtest']   #ndarray like list. (1536, 57)
ytest=mat['ytest']	#label   (1536, 1)

ZXtrain=preprocessing.scale(Xtrain)  #Z-normalization
ZXtest=preprocessing.scale(Xtest)  #将数据按期属性（按列axis=0(default),行axis=1）减去其均值，并处以其方差。

distance=np.zeros(shape=[1536,3065])
for rowtest in range(1536):
	for rowtrain in range(3065):
		d=(ZXtest[rowtest,:]-ZXtrain[rowtrain,:])**2
		distance[rowtest,rowtrain]=np.sqrt(sum(d))

K_list=list(range(1,10))+list(range(10,101,5))

errorrate_ZXtest=np.zeros(len(K_list))
error_column=0

for k in K_list:
	Ytest=np.zeros(shape=[1536,k])
	for row in range(1536):
		index=np.argsort(distance[row,:])
		for i in range(k):
			Ytest[row,i]=ytrain[index[i],0]

	ZXtest_p1=np.zeros(1536)
	ZXtest_p0=np.zeros(1536)
	for row in range(1536):
		counter=0
		for i in range(k):
			if Ytest[row,i]==1:
				counter+=1

		ZXtest_p1[row]=counter/k
		ZXtest_p0[row]=1-counter/k

	ZYtest=np.zeros(1536)
	for row in range(1536):
		if ZXtest_p1[row]>ZXtest_p0[row]:
			ZYtest[row]=1

	errorcounter=0
	for row in range(1536):
		if ZYtest[row]!=ytest[row]:
			errorcounter+=1
	errorrate_ZXtest[error_column]=errorcounter/1536

	if k==1: #0.095703125
		print(errorrate_ZXtest[error_column])

	if k==10: #0.0950520833333
		print(errorrate_ZXtest[error_column])

	if k==100:#0.127604166667
		print(errorrate_ZXtest[error_column])
	error_column+=1

plt.figure(1)
plt.plot(K_list,errorrate_ZXtest.T,color='b',label='test error rate')

#===========For train error=================%%

distance=np.zeros(shape=[3065,3065])
for rowtest in range(3065):
	for rowtrain in range(3065):
		d=(ZXtrain[rowtest,:]-ZXtrain[rowtrain,:])**2
		distance[rowtest,rowtrain]=np.sqrt(sum(d))

errorrate_ZXtrain=np.zeros(len(K_list))
error_column=0

for k in K_list:
	Ytrain=np.zeros(shape=[3065,k])
	for row in range(3065):
		index=np.argsort(distance[row,:])
		for i in range(k):
			Ytrain[row,i]=ytrain[index[i],0]

	Ztrain_p1=np.zeros(3065)
	Ztrain_p0=np.zeros(3065)
	for row in range(3065):
		counter=0
		for i in range(k):
			if Ytrain[row,i]==1:
				counter+=1

		Ztrain_p1[row]=counter/k
		Ztrain_p0[row]=1-counter/k

	ZYrain=np.zeros(3065)
	for row in range(3065):
		if Ztrain_p1[row]>Ztrain_p0[row]:
			ZYrain[row]=1

	errorcounter=0
	for row in range(3065):
		if ZYrain[row]!=ytrain[row]:
			errorcounter+=1
	errorrate_ZXtrain[error_column]=errorcounter/3065

	if k==1:#0.000652528548124
		print(errorrate_ZXtrain[error_column])

	if k==10: #0.0871125611746
		print(errorrate_ZXtrain[error_column])

	if k==100:#0.133115823817
		print(errorrate_ZXtrain[error_column])
	error_column+=1
plt.figure(1)
plt.plot(K_list,errorrate_ZXtrain,color='r',label='train error rate')
plt.title('KNN-Znormalization-errorrate')
plt.legend(loc=0)
plt.show()