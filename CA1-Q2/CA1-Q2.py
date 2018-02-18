import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn import preprocessing

mat = io.loadmat('spamData.mat')# <class 'dict'>
Xtrain=mat['Xtrain'] #feature  (3065, 57)
ytrain=mat['ytrain'] #<class 'numpy.ndarray'> (3065, 1)

Xtest=mat['Xtest']   #ndarray like list. (1536, 57)
ytest=mat['ytest']	#label   (1536, 1)

Nc=0
for row in ytrain:
	if row==1:
		Nc+=1
Pc=Nc/3065 #Prior probability

ZXtrain=preprocessing.scale(Xtrain)  #Z-normalization
ZXtest=preprocessing.scale(Xtest)  #将数据按期属性（按列axis=0(default),行axis=1）减去其均值，并处以其方差。

#----------------log transform---------------
LXtrain=np.log(Xtrain+0.1)
LXtest=np.log(Xtest+0.1)
#============devide a matrix to two matrixs============%%
#=====devide the ZXtrain into two matrix(zmc1 and zmc0)=====%
rowcounter1=0
rowcounter0=0
zmc1=np.zeros(shape=[Nc,57])
zmc0=np.zeros(shape=[3065-Nc,57])

for row in range(3065):
	if ytrain[row]==1:
		zmc1[rowcounter1]= ZXtrain[row]
		rowcounter1+=1
	else:
		zmc0[rowcounter0]=ZXtrain[row]
		rowcounter0+=1


#=====devide the LXtrain into two matrix(lmc1 and lmc0)=====%
lrowcounter1 = 0
lrowcounter0 = 0
lmc1=np.zeros(shape=[Nc,57])
lmc0=np.zeros(shape=[3065-Nc,57])
for row in range(3065):
	if ytrain[row]==1:
		lmc1[lrowcounter1]=LXtrain[row]
		lrowcounter1+=1
	else:
		lmc0[lrowcounter0]=LXtrain[row]
		lrowcounter0+=1

#=============calculate the error rate on Z-normalization ================%
#==== Testing Z-normalization=====%
tvar1=np.var(zmc1,0) #第二个参数为0，表示按列求方差
tvar0=np.var(zmc0,0)

tmean1=np.mean(zmc1,0)
tmean0=np.mean(zmc0,0)

sx=ZXtest.shape
zlogp1=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m1=0
	for column in range(sx[1]):
		m1+=(-0.5*(ZXtest[row,column]-tmean1[column])**2)/tvar1[column]+np.log(1/np.sqrt(2*np.pi*tvar1[column]))
	zlogp1[row]=np.log(Pc)+m1


zlogp0=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m0=0
	for column in range(sx[1]):
		m0+=(-0.5*(ZXtest[row,column]-tmean0[column])**2)/tvar0[column]+np.log(1/np.sqrt(2*np.pi*tvar0[column]))
	zlogp0[row]=np.log(1-Pc)+m0

#determine the class 1 or 0 by comparing
Yztest=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	if zlogp1[row]>zlogp0[row]:
		Yztest[row]=1
	else:
		Yztest[row]=0

#calculate the error rate for testing on Z-normalization

errorcounter=0
for row in range(sx[0]):
	if Yztest[row] != ytest[row]:
		errorcounter+=1

error_rate_test=errorcounter/1536

print('Testing error rate:',error_rate_test)

#==== Training Z-normalization=====%

tvar1=np.var(zmc1,0) #第二个参数为0，表示按列求方差
tvar0=np.var(zmc0,0)

tmean1=np.mean(zmc1,0)
tmean0=np.mean(zmc0,0)

sx=ZXtrain.shape
zlogp1=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m1=0
	for column in range(sx[1]):
		m1+=(-0.5*(ZXtrain[row,column]-tmean1[column])**2)/tvar1[column]+np.log(1/np.sqrt(2*np.pi*tvar1[column]))
	zlogp1[row]=np.log(Pc)+m1


zlogp0=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m0=0
	for column in range(sx[1]):
		m0+=(-0.5*(ZXtrain[row,column]-tmean0[column])**2)/tvar0[column]+np.log(1/np.sqrt(2*np.pi*tvar0[column]))
	zlogp0[row]=np.log(1-Pc)+m0

#determine the class 1 or 0 by comparing
Yztest=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	if zlogp1[row]>zlogp0[row]:
		Yztest[row]=1
	else:
		Yztest[row]=0

#calculate the error rate for testing on Z-normalization

errorcounter=0
for row in range(sx[0]):
	if Yztest[row] != ytrain[row]:
		errorcounter+=1

error_rate_ztrain=errorcounter/3065

print('Training error rate:',error_rate_ztrain)

#%% =============calculate the error rate on Log-transform ================%

#         %==== Testing Log-transform=====%

tvar1=np.var(lmc1,0) #第二个参数为0，表示按列求方差
tvar0=np.var(lmc0,0)

tmean1=np.mean(lmc1,0)
tmean0=np.mean(lmc0,0)

sx=LXtest.shape
llogp1=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m1=0
	for column in range(sx[1]):
		m1+=(-0.5*(LXtest[row,column]-tmean1[column])**2)/tvar1[column]+np.log(1/np.sqrt(2*np.pi*tvar1[column]))
	llogp1[row]=np.log(Pc)+m1


llogp0=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m0=0
	for column in range(sx[1]):
		m0+=(-0.5*(LXtest[row,column]-tmean0[column])**2)/tvar0[column]+np.log(1/np.sqrt(2*np.pi*tvar0[column]))
	llogp0[row]=np.log(1-Pc)+m0

#determine the class 1 or 0 by comparing
Yltest=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	if llogp1[row]>llogp0[row]:
		Yltest[row]=1
	else:
		Yltest[row]=0

#calculate the error rate for testing on Z-normalization

errorcounter=0
for row in range(sx[0]):
	if Yltest[row] != ytest[row]:
		errorcounter+=1

error_rate_ltest=errorcounter/1536

print('Testing error rate(Log):',error_rate_ltest)

#%==== Training Log-transform=====%


tvar1=np.var(lmc1,0) #第二个参数为0，表示按列求方差
tvar0=np.var(lmc0,0)

tmean1=np.mean(lmc1,0)
tmean0=np.mean(lmc0,0)

sx=LXtrain.shape
llogp1=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m1=0
	for column in range(sx[1]):
		m1+=(-0.5*(LXtrain[row,column]-tmean1[column])**2)/tvar1[column]+np.log(1/np.sqrt(2*np.pi*tvar1[column]))
	llogp1[row]=np.log(Pc)+m1


llogp0=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	m0=0
	for column in range(sx[1]):
		m0+=(-0.5*(LXtrain[row,column]-tmean0[column])**2)/tvar0[column]+np.log(1/np.sqrt(2*np.pi*tvar0[column]))
	llogp0[row]=np.log(1-Pc)+m0

#determine the class 1 or 0 by comparing
Yltest=np.zeros(shape=[sx[0],1])
for row in range(sx[0]):
	if llogp1[row]>llogp0[row]:
		Yltest[row]=1
	else:
		Yltest[row]=0

#calculate the error rate for testing on Z-normalization

errorcounter=0
for row in range(sx[0]):
	if Yltest[row] != ytrain[row]:
		errorcounter+=1

error_rate_ltrain=errorcounter/3065

print('Training error rate(Log):',error_rate_ltrain)






