import numpy as np
import myKmeans as mykmeans
from sklearn.metrics.pairwise import cosine_similarity
# Calculate the unit vector of our fix centroids first and only do the dot product of each coming data (patch)
# This will speed up the calculation once we have large numbers of centroids (features)
# In this way, we can get the score of each patch to each features.
class getbofs(object):

    def __init__(self , centroids):
        self.unitVec = []
        for item in centroids:
    #        print(item)
            item_norm = np.linalg.norm(item)
    #        print(item/item_norm)
            self.unitVec.append(item/item_norm)
     
        
    def getbof(self,data) :
        result = []
        for item in self.unitVec:
            result.append(np.dot(data,item))
    #        print(result)

        #return sum( cosine_similarity(data,centroids))
        return result