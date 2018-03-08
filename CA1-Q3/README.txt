####  Q3. Logistic regression  ####

This code is about Question 3. Our purpose is to sketch the plots of training and error rates and get the training and testing error rates for λ = 1, 10 and 100.


##Instructions to run code##
Just "Run" it in python3 and get the results.

##Attention##
In log and binarization procession, use logistic.cdf(x) instead of sigmoid() to avoid overflow.
And use np.linalg.pinv(HH)*greg to avoid singular matrix. 