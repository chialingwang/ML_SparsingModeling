from my_vlad import my_vlad
from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search
import numpy as np
import image_norm_test as myData;
import ruleOfThumb
from my_svm import my_svm,stop

def run(centroid , group_num , train_X , train_y ,test_X , test_y, method='knn' , n_nb = 2 , seed = 371986):
    
    # will load data as the patch size defined , 3 means 3*3 = 9 for each patch, and will return the dictionary included:
    # 'data'  (one patch)  , 'target' (the sample of this patch belongs to ) , 'filename' (the file comes from)
    bofs = []
    vlad = my_vlad(centroid[group_num])
    for file in train_X:
        mydata,y = myData.load_sig_data(file , 3)
        bofs.append(vlad.get_vlad(mydata['data']).flatten() )
        
    #knn_init = KNeighborsClassifier()
    #parameters = {'n_neighbors':[ 5, 10 , 15]}
    #knn = grid_search.GridSearchCV(knn_init, parameters)

    bofs_test = []
    for file in test_X:
        mydata,y = myData.load_sig_data(file , 3)
        bofs_test.append(vlad.get_vlad(mydata['data']).flatten() )

    if(method == "knn"):
        knn = KNeighborsClassifier(n_neighbors = n_nb)

        knn.fit(bofs, train_y)
        predicted = knn.predict(bofs_test)

        score = knn.score(bofs_test,test_y)
        
    if(method == "LinearSVM"):
        svm = my_svm(iter = 100 )
        w , obj , sp = svm.my_sgd(bofs,train_y, seed = seed , stop = stop.perfor , step = 0.11, t0 = 2, C = 1  )
        score = 100-svm.predict(bofs_test,test_y)
            
    return score   