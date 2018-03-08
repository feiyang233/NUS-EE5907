import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from scipy.stats import logistic

mat = io.loadmat('spamData.mat')# <class 'dict'>
Xtrain=mat['Xtrain'] #feature  (3065, 57)
ytrain=mat['ytrain'] #<class 'numpy.ndarray'> (3065, 1)

Xtest=mat['Xtest']   #ndarray like list. (1536, 57)
ytest=mat['ytest']	#label   (1536, 1)

column_train=np.ones(3065)
column_test=np.ones(1536)
 #将数据按期属性（按列axis=0(default),行axis=1）减去其均值，并处以其方差。
LXtrain1=np.log(Xtrain+0.1)
LXtest1=np.log(Xtest+0.1)
LXtrain=np.insert(LXtrain1, 0, values=column_train, axis=1)
LXtest=np.insert(LXtest1, 0, values=column_test, axis=1)
#%% =============calculate the error rate on Z-normalization ================%
#%%===== test error rate on Z-normalization =====%%

Ytest_column = 0
lmd_list=list(range(1,10))+list(range(10,101,5))
error_rate_Ltest=np.zeros(len(lmd_list))
for lmd in lmd_list:
	#print(lmd)
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
			x=w.T.dot(LXtrain[row,:])
			mu[row,0]=logistic.cdf(x)
			mu_y[row,0]=mu[row,0]-ytrain[row,0]
			S[row,row]=mu[row,0]*(1-mu[row,0])
		
		for counter in range(1,58):
			lmdw[counter,0]=lmd*w[counter,0]

		lmdw[0,0]=w[0,0]
		greg=LXtrain.T.dot(mu_y)+lmdw
		Hreg=LXtrain.T.dot(S).dot(LXtrain)+lmd*I
		HH=np.mat(Hreg)
		try:
			convergence=(HH.I)*greg
		except np.linalg.linalg.LinAlgError:
			convergence=np.linalg.pinv(HH)*greg
		w=w-convergence	
		margin=(convergence).T*(convergence)
#======calculate the error rate and plot for LXtest  
  # Classify the LXtest to class 1 or class 0   

	p=np.zeros(shape=[1536,1])
	ZYtest=np.zeros(1536)
	for row in range(1536):
		p[row]=w.T.dot(LXtest[row])
		if p[row]>0:
			ZYtest[row]=1

	errorcounter=0
	for row in range(1536):
		if ZYtest[row]!=ytest[row]:
			errorcounter+=1

	error_rate_Ltest[Ytest_column]=errorcounter/1536

	if lmd==1: #0.052734375
		print('Z_test')
		print(error_rate_Ltest[Ytest_column])
	if lmd==10: #0.0520833333333
		print(error_rate_Ltest[Ytest_column])
	if lmd==100: # 0.06640625
		print(error_rate_Ltest[Ytest_column])

	Ytest_column+=1

#np.save("error_rate_Ltest.npy",error_rate_Ltest)

plt.figure(1)
plt.plot(lmd_list,error_rate_Ltest,color='b',label='test error rate')


#=============calculate the error rate on Z-normalization ================%
#===== training error rate on Z-normalization =====%%
Ytrain_column = 0
error_rate_Ltrain=np.zeros(len(lmd_list))

for lmd in lmd_list:
	#print(lmd)
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
			x=w.T.dot(LXtrain[row,:])
			mu[row,0]=logistic.cdf(x)
			mu_y[row,0]=mu[row,0]-ytrain[row,0]
			S[row,row]=mu[row,0]*(1-mu[row,0])
		
		for counter in range(1,58):
			lmdw[counter,0]=lmd*w[counter,0]

		lmdw[0,0]=w[0,0]
		greg=LXtrain.T.dot(mu_y)+lmdw
		Hreg=LXtrain.T.dot(S).dot(LXtrain)+lmd*I
		HH=np.mat(Hreg)
		try:
			convergence=(HH.I)*greg
		except np.linalg.linalg.LinAlgError:
			convergence=np.linalg.pinv(HH)*greg
		w=w-convergence	
		
		margin=(convergence).T*(convergence)
		
#======calculate the error rate and plot for LXtest  
  # Classify the xtrain to class 1 or class 0   

	p=np.zeros(shape=[3065,1])
	ZYtrain=np.zeros(3065)
	for row in range(3065):
		p[row]=w.T.dot(LXtrain[row])
		if p[row]>0:
			ZYtrain[row]=1

	errorcounter=0
	for row in range(3065):
		if ZYtrain[row]!=ytrain[row]:
			errorcounter+=1

	error_rate_Ltrain[Ytrain_column]=errorcounter/3065

	if lmd==1: #0.0642740619902
		print('Z_train')
		print(error_rate_Ltrain[Ytrain_column])
	if lmd==10: # 0.0672104404568
		print(error_rate_Ltrain[Ytrain_column])
	if lmd==100: # 0.0936378466558
		print(error_rate_Ltrain[Ytrain_column])

	Ytrain_column+=1
#np.save("error_rate_Ltrain.npy",error_rate_Ltrain)

plt.figure(1)
plt.plot(lmd_list,error_rate_Ltrain,color='r',label='train error rate')
plt.title('error-rate-Log-normalization')
plt.legend(loc=0)
plt.show()







