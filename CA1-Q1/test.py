import matplotlib.pyplot as plt
import numpy as np
errortest=np.load("errortest.npy")
errortrain=np.load("errortrain.npy")
a=np.arange(0,100.5,0.5)
plt.figure(1)
plt.plot(a,errortest.T,color='b',label='test error rate')



plt.figure(1)
plt.plot(a,errortrain.T,color='r',label='train error rate')
plt.title('Error rate -- Beta-bernoulli Naive Bayes')
plt.legend(loc=0)
plt.show()


