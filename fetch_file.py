import os, sys
import image_norm_test as myData;
import numpy as np
from sklearn.cross_validation import train_test_split
seed = 3071986   # set random seed
rng = np.random.RandomState(seed)
import re

def retrive_lable(filename):
    new_name = filename.split('\\')[2].split('_')[0]
    sample = int(re.findall(r'\d+', new_name)[0])
    return sample
    
def fetchFile(accessPath , numOfimageOfeachSample = 91):
    filelist = []
    lable = []
    index = 0
    i = 0
    for file in os.listdir(accessPath):
        if(index % 92 == i):
            completeName = os.path.join(accessPath, file)  
            filelist.append(completeName)
            lable.append(retrive_lable(completeName))
            if(i < numOfimageOfeachSample-1):  
                i+=1
            else:
                i = 0
        index +=1
    return filelist , lable