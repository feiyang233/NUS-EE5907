import matplotlib.pyplot as plt
import numpy as np
errortest=np.load("errortest.npy")
errortrain=np.load("errortrain.npy")
plt.figure(1)
a=np.arange(0,100.5,0.5)
plt.scatter(a,errortest.T,marker='o',color='g', label='test error rate')
plt.figure(2)
plt.plot(a,errortest.T,color='b',label='test error rate')


plt.figure(1)
plt.scatter(a,errortrain.T,marker='o',color='r',label='train error rate')
plt.title('Error rate -- Beta-bernoulli Naive Bayes')

plt.legend(loc=0)
plt.figure(2)
plt.plot(a,errortrain.T,color='r',label='train error rate')
plt.title('Error rate -- Beta-bernoulli Naive Bayes')
plt.legend(loc=0)
plt.show()


