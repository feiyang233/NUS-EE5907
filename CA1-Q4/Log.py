import numpy as np
from scipy import io
import matplotlib.pyplot as plt

mat = io.loadmat('spamData.mat')# <class 'dict'>
Xtrain=mat['Xtrain'] #feature  (3065, 57)
ytrain=mat['ytrain'] #<class 'numpy.ndarray'> (3065, 1)

Xtest=mat['Xtest']   #ndarray like list. (1536, 57)
ytest=mat['ytest']	#label   (1536, 1)

LXtrain=np.log(Xtrain+0.1)
LXtest=np.log(Xtest+0.1)
distance=np.zeros(shape=[1536,3065])

for rowtest in range(1536):
	for rowtrain in range(3065):
		d=(LXtest[rowtest,:]-LXtrain[rowtrain,:])**2
		distance[rowtest,rowtrain]=np.sqrt(sum(d))

K_list=list(range(1,10))+list(range(10,101,5))

errorrate_LXtest=np.zeros(len(K_list))
error_column=0

for k in K_list:
	Ytest=np.zeros(shape=[1536,k])
	for row in range(1536):
		index=np.argsort(distance[row,:])
		for i in range(k):
			Ytest[row,i]=ytrain[index[i],0]

	LXtest_p1=np.zeros(1536)
	LXtest_p0=np.zeros(1536)
	for row in range(1536):
		counter=0
		for i in range(k):
			if Ytest[row,i]==1:
				counter+=1

		LXtest_p1[row]=counter/k
		LXtest_p0[row]=1-counter/k

	BYtest=np.zeros(1536)
	for row in range(1536):
		if LXtest_p1[row]>LXtest_p0[row]:
			BYtest[row]=1

	errorcounter=0
	for row in range(1536):
		if BYtest[row]!=ytest[row]:
			errorcounter+=1
	errorrate_LXtest[error_column]=errorcounter/1536

	if k==1: #0.0611979166667
		print(errorrate_LXtest[error_column])

	if k==10: #0.0579427083333
		print(errorrate_LXtest[error_column])

	if k==100:#0.0885416666667
		print(errorrate_LXtest[error_column])
	error_column+=1

plt.figure(1)
plt.plot(K_list,errorrate_LXtest.T,color='b',label='test error rate')

#===========For train error=================%%

distance=np.zeros(shape=[3065,3065])
for rowtest in range(3065):
	for rowtrain in range(3065):
		d=(LXtrain[rowtest,:]-LXtrain[rowtrain,:])**2
		distance[rowtest,rowtrain]=np.sqrt(sum(d))

errorrate_LXtrain=np.zeros(len(K_list))
error_column=0

for k in K_list:
	Ytrain=np.zeros(shape=[3065,k])
	for row in range(3065):
		index=np.argsort(distance[row,:])
		for i in range(k):
			Ytrain[row,i]=ytrain[index[i],0]

	Ltrain_p1=np.zeros(3065)
	Ltrain_p0=np.zeros(3065)
	for row in range(3065):
		counter=0
		for i in range(k):
			if Ytrain[row,i]==1:
				counter+=1

		Ltrain_p1[row]=counter/k
		Ltrain_p0[row]=1-counter/k

	LYrain=np.zeros(3065)
	for row in range(3065):
		if Ltrain_p1[row]>Ltrain_p0[row]:
			LYrain[row]=1

	errorcounter=0
	for row in range(3065):
		if LYrain[row]!=ytrain[row]:
			errorcounter+=1
	errorrate_LXtrain[error_column]=errorcounter/3065

	if k==1: #0.000652528548124
		print(errorrate_LXtrain[error_column])

	if k==10: #0.0548123980424
		print(errorrate_LXtrain[error_column])

	if k==100:#0.089722675367
		print(errorrate_LXtrain[error_column])
	error_column+=1
plt.figure(1)
plt.plot(K_list,errorrate_LXtrain,color='r',label='train error rate')
plt.title('KNN-Log transform-errorrate')
plt.legend(loc=0)
plt.show()





