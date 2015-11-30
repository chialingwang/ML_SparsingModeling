import numpy as np
import sys,os
import time
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import data_gen
from my_svm import my_svm,stop

def ruleOfThumb(n = 10**5 , dim = 2 , C = 1 , seed = 371986 ):
    rng = np.random.RandomState(seed)
    Cov = rng.normal(0, 1, (dim, dim))
    XL, yL = data_gen.dataset_fixed_cov(n, dim, seed , Cov)
    rng = np.random.RandomState(seed)
    min_error = sys.maxsize
    permutation = rng.permutation(len(ＸL))

    #print(X , y)
    XL, yL = XL[permutation], yL[permutation]

    XL1 = XL
    yL1 = yL
    n_features = XL1.shape[1]
    #print(n_features)
    # split 50%-50% training sets and test sets
    scaler = preprocessing.StandardScaler()
    XL1 = scaler.fit_transform(XL1,yL1)
    #test_X1 = scaler.fit_transform(test_X1,test_y1)

    train_XL1, test_XL1, train_yL1, test_yL1 = train_test_split(XL1, yL1, train_size=0.5, random_state=seed)
    
    
    svm = my_svm(iter = 100 )
    start_time = time.time()
    w , obj , step= svm.my_gradient_descent(train_XL1,train_yL1, stop = stop.perfor , C = C )
    print("---Cost %s seconds for GD ---" % (time.time() - start_time))
    error_GD = svm.predict(test_XL1,test_yL1,w=w[len(w)-1])
    svm = my_svm(iter = 100 )
    start_time = time.time()
    w , obj , sp = svm.my_sgd(train_XL1,train_yL1, seed = seed , stop = stop.perfor , step = 0.11, t0 = 2, C = C  )
    print("---Cost %s seconds for SGD ---" % (time.time() - start_time))
    error_SGD = svm.predict(test_XL1,test_yL1,w=w[len(w)-1])
    if(error_GD < error_SGD):
        print("Use GD to run the data number = %f and dimention = %d , and C = %d , with error rate = %f " %(n , dim , C , error_GD))
    else :
        print("Use SGD to run the data number = %f and dimention = %d , and C = %d , with error rate = %f" %(n , dim , C , error_SGD))