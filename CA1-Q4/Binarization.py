import numpy as np
from scipy import io
import matplotlib.pyplot as plt

mat = io.loadmat('spamData.mat')# <class 'dict'>
Xtrain=mat['Xtrain'] #feature  (3065, 57)
ytrain=mat['ytrain'] #<class 'numpy.ndarray'> (3065, 1)

Xtest=mat['Xtest']   #ndarray like list. (1536, 57)
ytest=mat['ytest']	#label   (1536, 1)

def binarization(X):
	lis= np.zeros(X.shape)
	n=0
	for row in X:
		m=0
		for k in row:
			if k!=0:
				lis[n,m]=1
			m+=1
		n+=1
	return lis

BXtrain=binarization(Xtrain) #data pre process
BXtest=binarization(Xtest)

distance=np.zeros(shape=[1536,3065])

for rowtest in range(1536):
	for rowtrain in range(3065):
		d=BXtest[rowtest,:]-BXtrain[rowtrain,:]
		dis=len(np.nonzero(d)[0])
		distance[rowtest,rowtrain]=dis

K_list=list(range(1,10))+list(range(10,101,5))

errorrate_BXtest=np.zeros(len(K_list))
error_column=0

for k in K_list:
	Ytest=np.zeros(shape=[1536,k])
	for row in range(1536):
		index=np.argsort(distance[row,:])
		for i in range(k):
			Ytest[row,i]=ytrain[index[i],0]

	BXtest_p1=np.zeros(1536)
	BXtest_p0=np.zeros(1536)
	for row in range(1536):
		counter=0
		for i in range(k):
			if Ytest[row,i]==1:
				counter+=1

		BXtest_p1[row]=counter/k
		BXtest_p0[row]=1-counter/k

	BYtest=np.zeros(1536)
	for row in range(1536):
		if BXtest_p1[row]>BXtest_p0[row]:
			BYtest[row]=1

	errorcounter=0
	for row in range(1536):
		if BYtest[row]!=ytest[row]:
			errorcounter+=1
	errorrate_BXtest[error_column]=errorcounter/1536

	if k==1: #0.0729166666667
		print(errorrate_BXtest[error_column])

	if k==10: #0.0833333333333
		print(errorrate_BXtest[error_column])

	if k==100:#0.0930989583333
		print(errorrate_BXtest[error_column])
	error_column+=1

plt.figure(1)
plt.plot(K_list,errorrate_BXtest.T,color='b',label='test error rate')

#===========For train error=================%%
distance=np.zeros(shape=[3065,3065])
for rowtest in range(3065):
	for rowtrain in range(3065):
		d=BXtrain[rowtest,:]-BXtrain[rowtrain,:]
		dis=len(np.nonzero(d)[0])
		distance[rowtest,rowtrain]=dis

errorrate_BXtrain=np.zeros(len(K_list))
error_column=0



for k in K_list:
	Ytrain=np.zeros(shape=[3065,k])
	for row in range(3065):
		index=np.argsort(distance[row,:])
		for i in range(k):
			Ytrain[row,i]=ytrain[index[i],0]

	Btrain_p1=np.zeros(3065)
	Btrain_p0=np.zeros(3065)
	for row in range(3065):
		counter=0
		for i in range(k):
			if Ytrain[row,i]==1:
				counter+=1

		Btrain_p1[row]=counter/k
		Btrain_p0[row]=1-counter/k

	BYrain=np.zeros(3065)
	for row in range(3065):
		if Btrain_p1[row]>Btrain_p0[row]:
			BYrain[row]=1

	errorcounter=0
	for row in range(3065):
		if BYrain[row]!=ytrain[row]:
			errorcounter+=1
	errorrate_BXtrain[error_column]=errorcounter/3065

	if k==1: #0.010766721044
		print(errorrate_BXtrain[error_column])

	if k==10: #0.0721044045677
		print(errorrate_BXtrain[error_column])

	if k==100:#0.109298531811
		print(errorrate_BXtrain[error_column])
	error_column+=1
plt.figure(1)
plt.plot(K_list,errorrate_BXtrain,color='r',label='train error rate')
plt.title('KNN-Binarization-errorrate')
plt.legend(loc=0)
plt.show()








