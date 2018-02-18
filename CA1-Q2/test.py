import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn import preprocessing


a = np.array(range(15),dtype=np.float64).reshape(3,5)
c=np.zeros(shape=[3,5])

amean=np.mean(a,0)
std1=np.std(a,0)


for i in range(3):
	for j in range(5):
		c[i,j]=(a[i,j]-amean[j])/std1[j]

b=preprocessing.scale(a)
sx=a.shape
ii=9
print(ii**2)
