import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn import preprocessing


def sigmoid(x):
  return 1 / (1 + np.exp(-x)) #np.exp

mat = io.loadmat('spamData.mat')# <class 'dict'>
Xtrain=mat['Xtrain'] #feature  (3065, 57)
ytrain=mat['ytrain'] #<class 'numpy.ndarray'> (3065, 1)

Xtest=mat['Xtest']   #ndarray like list. (1536, 57)
ytest=mat['ytest']	#label   (1536, 1)

column_train=np.ones(3065)
column_test=np.ones(1536)
ZXtrain1=preprocessing.scale(Xtrain)  #Z-normalization
ZXtest1=preprocessing.scale(Xtest)  #将数据按期属性（按列axis=0(default),行axis=1）减去其均值，并处以其方差。
ZXtrain=np.insert(ZXtrain1, 0, values=column_train, axis=1)
ZXtest=np.insert(ZXtest1, 0, values=column_test, axis=1)
#%% =============calculate the error rate on Z-normalization ================%
#%%===== test error rate on Z-normalization =====%%

Ytest_column = 0
lmd_list=list(range(1,10))+list(range(10,101,5))
error_rate_Ztest=np.zeros(len(lmd_list))
for lmd in lmd_list:
	w=np.zeros(shape=[58,1])
	I=np.eye(58)
	margin=1
	lmdw=np.zeros(shape=[58,1])
	mu=np.zeros(shape=[3065,1])
	mu_y=np.zeros(shape=[3065,1])
	S=np.eye(3065)
	#=====Using Newton's Method until convergence
	while margin>10**-3:
		
		for row in range(3065):
			x=w.T.dot(ZXtrain[row,:])
			mu[row,0]=sigmoid(x)
			mu_y[row,0]=mu[row,0]-ytrain[row,0]
			S[row,row]=mu[row,0]*(1-mu[row,0])
		
		for counter in range(1,58):
			lmdw[counter,0]=lmd*w[counter,0]

		lmdw[0,0]=w[0,0]
		greg=ZXtrain.T.dot(mu_y)+lmdw
		Hreg=ZXtrain.T.dot(S).dot(ZXtrain)+lmd*I
		HH=np.mat(Hreg)
		convergence=(HH.I)*greg
		w=w-convergence	
		
		margin=(convergence).T*(convergence)
		
#======calculate the error rate and plot for ZXtest  
  # Classify the LXtest to class 1 or class 0   

	p=np.zeros(shape=[1536,1])
	ZYtest=np.zeros(1536)
	for row in range(1536):
		p[row]=w.T.dot(ZXtest[row])
		if p[row]>0:
			ZYtest[row]=1

	errorcounter=0
	for row in range(1536):
		if ZYtest[row]!=ytest[row]:
			errorcounter+=1

	error_rate_Ztest[Ytest_column]=errorcounter/1536

	if lmd==1: #0.068359375
		print('Z_test')
		print(error_rate_Ztest[Ytest_column])
	if lmd==10: #0.0716145833333
		print(error_rate_Ztest[Ytest_column])
	if lmd==100: # 0.0852864583333
		print(error_rate_Ztest[Ytest_column])

	Ytest_column+=1
#np.save("error_rate_Ztest.npy",error_rate_Ztest)

plt.figure(1)
plt.plot(lmd_list,error_rate_Ztest.T,color='r',label='test error rate')


#=============calculate the error rate on Z-normalization ================%
#===== training error rate on Z-normalization =====%%
Ytrain_column = 0
error_rate_Ztrain=np.zeros(len(lmd_list))
for lmd in lmd_list:
	w=np.zeros(shape=[58,1])
	I=np.eye(58)
	margin=1
	lmdw=np.zeros(shape=[58,1])
	mu=np.zeros(shape=[3065,1])
	mu_y=np.zeros(shape=[3065,1])
	S=np.eye(3065)
	#=====Using Newton's Method until convergence
	while margin>10**-3:
		
		for row in range(3065):
			x=w.T.dot(ZXtrain[row,:])
			mu[row,0]=sigmoid(x)
			mu_y[row,0]=mu[row,0]-ytrain[row,0]
			S[row,row]=mu[row,0]*(1-mu[row,0])
		
		for counter in range(1,58):
			lmdw[counter,0]=lmd*w[counter,0]

		lmdw[0,0]=w[0,0]
		greg=ZXtrain.T.dot(mu_y)+lmdw
		Hreg=ZXtrain.T.dot(S).dot(ZXtrain)+lmd*I
		HH=np.mat(Hreg)
		convergence=(HH.I)*greg
		w=w-convergence	
		
		margin=(convergence).T*(convergence)
		
#======calculate the error rate and plot for ZXtest  
  # Classify the xtrain to class 1 or class 0   

	p=np.zeros(shape=[3065,1])
	ZYtrain=np.zeros(3065)
	for row in range(3065):
		p[row]=w.T.dot(ZXtrain[row])
		if p[row]>0:
			ZYtrain[row]=1

	errorcounter=0
	for row in range(3065):
		if ZYtrain[row]!=ytrain[row]:
			errorcounter+=1

	error_rate_Ztrain[Ytrain_column]=errorcounter/3065

	if lmd==1: #0.0763458401305
		print('Z_train')
		print(error_rate_Ztrain[Ytrain_column])
	if lmd==10: # 0.0822185970636
		print(error_rate_Ztrain[Ytrain_column])
	if lmd==100: # 0.0965742251223
		print(error_rate_Ztrain[Ytrain_column])

	Ytrain_column+=1
#np.save("error_rate_Ztrain.npy",error_rate_Ztrain)



plt.figure(1)
plt.plot(lmd_list,error_rate_Ztrain.T,color='g',label='train error rate')
plt.title('error-rate-Z-normalization')
plt.legend(loc=0)
plt.show()







