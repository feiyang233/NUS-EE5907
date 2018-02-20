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

def thetajc(a):
	
	thetajc = np.zeros(shape = (2,57))

	for column in range(57): #j=1 spam
		counterc1 = 0
		counterj1 = 0
		for row in range(3065):
			if ytrain[row,0]==1:
				counterc1+=1
				if BXtrain[row,column]==1:
					counterj1+=1
		thetajc[0,column]=(counterj1+a)/(counterc1+2*a)

	for column in range(57):
		counterc0 = 0
		counterj0 = 0
		for row in range(3065):
			if ytrain[row,0]==0: #j=0 not spam
				counterc0+=1
				if BXtrain[row,column]==1:
					counterj0+=1
		thetajc[1,column]=(counterj0+a)/(counterc0+2*a)



	return thetajc


Nc=0
for row in ytrain:
	if row==1:
		Nc+=1
Pc=Nc/3065 #Prior probability


#For testing error
errortest=np.zeros(shape=(1,201))
a=np.arange(0,100.5,0.5)
Ytestcolumn=0
for i in a:
	Thetajc=thetajc(i)
	logp=np.zeros(shape=(1536,2))
	Yresult=np.zeros(shape=(1536,1))

	for row in range(1536): #probability of judging them to class 1
		 mc1=0
		 for column in range(57):
		 	xj=BXtest[row,column]
		 	if xj==1:
		 		mc1+=np.log(Thetajc[0,column])
		 	else:
		 		mc1+=np.log(1-Thetajc[0,column])

		 	logp[row,0]=np.log(Pc)+mc1

	for row in range(1536):  #probability of judging them to class 0
		 mc0=0
		 for column in range(57):
		 	xj=BXtest[row,column]
		 	if xj==1:
		 		mc0+=np.log(Thetajc[1,column])
		 	else:
		 		mc0+=np.log(1-Thetajc[1,column])

		 	logp[row,1]=np.log(Pc)+mc0

	for row in range(1536): #determine the class 0/1 by comparing
		if logp[row,0]>logp[row,1]:
			Yresult[row,0]=1
		else:
			Yresult[row,0]=0


	errorcounter=0
	for row in range(1536):
		if Yresult[row,0]!=ytest[row,0]:
			errorcounter+=1

	errortest[0,Ytestcolumn]=errorcounter/1536

	if i==1:
		print(errortest[0,Ytestcolumn])
	if i==10:
		print(errortest[0,Ytestcolumn])
	if i==100:
		print(errortest[0,Ytestcolumn])
	Ytestcolumn+=1
	



#np.save("errortest.npy",errortest) #0.109375 0.11328125  0.126953125

a=np.arange(0,100.5,0.5)
plt.figure(1)
plt.plot(a,errortest.T,color='b',label='test error rate')


#For traing error
errortrain=np.zeros(shape=(1,201))
Ytestcolumn=0
a=np.arange(0,100.5,0.5)
for i in a:
	Thetajc=thetajc(i)
	logp=np.zeros(shape=(3065,2))
	Yresult=np.zeros(shape=(3065,1))

	for row in range(3065): #probability of judging them to class 1
		 mc1=0
		 for column in range(57):
		 	xj=BXtrain[row,column]
		 	if xj==1:
		 		mc1+=np.log(Thetajc[0,column])
		 	else:
		 		mc1+=np.log(1-Thetajc[0,column])

		 	logp[row,0]=np.log(Pc)+mc1

	for row in range(3065):  #probability of judging them to class 0
		 mc0=0
		 for column in range(57):
		 	xj=BXtrain[row,column]
		 	if xj==1:
		 		mc0+=np.log(Thetajc[1,column])
		 	else:
		 		mc0+=np.log(1-Thetajc[1,column])

		 	logp[row,1]=np.log(Pc)+mc0

	for row in range(3065): #determine the class 0/1 by comparing
		if logp[row,0]>logp[row,1]:
			Yresult[row,0]=1
		else:
			Yresult[row,0]=0


	errorcounter=0
	for row in range(3065):
		if Yresult[row,0]!=ytrain[row,0]:
			errorcounter+=1

	errortrain[0,Ytestcolumn]=errorcounter/3065

	if i==1:
		print(errortrain[0,Ytestcolumn])
	if i==10:
		print(errortrain[0,Ytestcolumn])
	if i==100:
		print(errortrain[0,Ytestcolumn])
	Ytestcolumn+=1

#np.save("errortrain.npy",errortrain) #0.109951060359  0.114192495922  0.138336052202

plt.figure(1)
plt.plot(a,errortrain.T,color='r',label='train error rate')
plt.title('Error rate -- Beta-bernoulli Naive Bayes')
plt.legend(loc=0)
plt.show()










