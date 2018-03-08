import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from scipy.stats import logistic

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

BXtrain1=binarization(Xtrain) #data pre process
BXtest1=binarization(Xtest)

column_train=np.ones(3065)
column_test=np.ones(1536)

BXtrain=np.insert(BXtrain1, 0, values=column_train, axis=1)
BXtest=np.insert(BXtest1, 0, values=column_test, axis=1)
#%% =============calculate the error rate on Z-normalization ================%
#%%===== test error rate on Z-normalization =====%%

Ytest_column = 0
lmd_list=list(range(1,10))+list(range(10,101,5))

error_rate_Btest=np.zeros(len(lmd_list))

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
			x=w.T.dot(BXtrain[row,:])
			mu[row,0]=logistic.cdf(x)
			mu_y[row,0]=mu[row,0]-ytrain[row,0]
			S[row,row]=mu[row,0]*(1-mu[row,0])
		
		for counter in range(1,58):
			lmdw[counter,0]=lmd*w[counter,0]

		lmdw[0,0]=w[0,0]
		greg=BXtrain.T.dot(mu_y)+lmdw
		Hreg=BXtrain.T.dot(S).dot(BXtrain)+lmd*I
		HH=np.mat(Hreg)
		try:
			convergence=(HH.I)*greg
		except np.linalg.linalg.LinAlgError:
			convergence=np.linalg.pinv(HH)*greg

		w=w-convergence
		
		margin=(convergence).T*(convergence)
		
#======calculate the error rate and plot for BXtrain  
  # Classify the LXtest to class 1 or class 0   

	p=np.zeros(shape=[1536,1])
	BYtest=np.zeros(1536)
	for row in range(1536):
		p[row]=w.T.dot(BXtest[row])
		if p[row]>0:
			BYtest[row]=1

	errorcounter=0
	for row in range(1536):
		if BYtest[row]!=ytest[row]:
			errorcounter+=1

	error_rate_Btest[Ytest_column]=errorcounter/1536

	if lmd==1: #0.0735677083333
		print('B_test')
		print(error_rate_Btest[Ytest_column])
	if lmd==10: #0.0755208333333
		print(error_rate_Btest[Ytest_column])
	if lmd==100: # 0.0963541666667
		print(error_rate_Btest[Ytest_column])

	Ytest_column+=1
#np.save("error_rate_Btest.npy",error_rate_Btest)

plt.figure(1)
plt.plot(lmd_list,error_rate_Btest,color='r',label='B-test error rate')


#=============calculate the error rate on Z-normalization ================%
#===== training error rate on Z-normalization =====%%
Ytrain_column = 0
error_rate_Btrain=np.zeros(len(lmd_list))
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
			x=w.T.dot(BXtrain[row,:])
			mu[row,0]=logistic.cdf(x)
			mu_y[row,0]=mu[row,0]-ytrain[row,0]
			S[row,row]=mu[row,0]*(1-mu[row,0])
		
		for counter in range(1,58):
			lmdw[counter,0]=lmd*w[counter,0]

		lmdw[0,0]=w[0,0]
		greg=BXtrain.T.dot(mu_y)+lmdw
		Hreg=BXtrain.T.dot(S).dot(BXtrain)+lmd*I
		HH=np.mat(Hreg)
		try:
			convergence=(HH.I)*greg
		except np.linalg.linalg.LinAlgError:
			convergence=np.linalg.pinv(HH)*greg

		w=w-convergence	
		margin=(convergence).T*(convergence)
		
#======calculate the error rate and plot for BXtrain  
  # Classify the xtrain to class 1 or class 0   

	p=np.zeros(shape=[3065,1])
	BYtrain=np.zeros(3065)
	for row in range(3065):
		p[row]=w.T.dot(BXtrain[row])
		if p[row]>0:
			BYtrain[row]=1

	errorcounter=0
	for row in range(3065):
		if BYtrain[row]!=ytrain[row]:
			errorcounter+=1

	error_rate_Btrain[Ytrain_column]=errorcounter/3065

	if lmd==1: #0.0642740619902
		print('B_train')
		print(error_rate_Btrain[Ytrain_column])
	if lmd==10: # 0.0672104404568
		print(error_rate_Btrain[Ytrain_column])
	if lmd==100: # 0.0936378466558
		print(error_rate_Btrain[Ytrain_column])

	Ytrain_column+=1
#np.save("error_rate_Btrain.npy",error_rate_Btrain)



plt.figure(1)
plt.plot(lmd_list,error_rate_Btrain,color='g',label='B-train error rate')
plt.title('error-rate-Binarization')
plt.legend(loc=0)
plt.show()







