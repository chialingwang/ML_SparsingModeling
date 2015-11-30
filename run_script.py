import os, sys
import image_norm_test as myData;
import numpy as np
from sklearn.cross_validation import train_test_split
seed = 3071986   # set random seed
rng = np.random.RandomState(seed)
import re
import fetch_file

###################################################
# fetch all the file in the directory path  defined#
pwd = os.pardir;
accessPath = r"%s\patch_database" %pwd;
filelist = []
print(accessPath)


# in this way we can extract only certain number image of each samples, ex. 2 means only 2 images of one texture
filelist , lable = fetch_file.fetchFile(accessPath , 4)
print(filelist[0])
print(filelist[100])
filelist = np.array(filelist).transpose()[0:100];
lable = np.array(lable).transpose()[0:100];
print(lable)

###################################################
# permutation data and split into train and test #
from sklearn import preprocessing


rng = np.random.RandomState(seed)
permutation = rng.permutation(len(filelist))

#print(permutation)
#print(filelist[permutation])
X, y = filelist[permutation], lable[permutation]


# split our data into half of train data and half of text data randomly.
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, random_state=seed)


print(train_y)
print(test_y)

###################################################
#Start to get the features learned from feed in data
import get_feature

centroid = []
count = 0
temp = []
n = 3*3
num_data = 10000
clusters = [16,32,64,128]
#cluster = 16
#centroid_file = r"%s\%dfeature_%d.txt" %(pwd ,  cluster , num_data);
for cluster in clusters:
    centroid_file = r"%s\25_4_3_%d.txt" %(pwd , cluster)
    centroid.append(get_feature.read_features(centroid_file))
'''
data = np.loadtxt(centroid_file,delimiter="\n")
for i in range(0 , len(data)):
    temp.append(data[i])
    count += 1
    if(count == n):
        centroid.append(temp)
        temp = []
        count = 0
'''
print(centroid[0])
print(len(centroid[0]))

###################################################
# Final Run #####
import run

run.run(centroid , 0 , train_X , train_y ,test_X , test_y, method='knn' , n_nb = 2)