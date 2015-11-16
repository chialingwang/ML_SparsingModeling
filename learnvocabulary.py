import numpy as np
import myKmeans as mykmeans
from sklearn.metrics.pairwise import cosine_similarity


# Use only the sample size of datathat feed in  (randomly choose) to do kmeans cluster
def learnvocabulary(data, sample_size ,rng , cluster_num = 100 , T = 100):
    choose_data = ramdon_data(data, sample_size ,rng )
    clusts,centroid,lable = mykmeans.kmeansopt(choose_data, cluster_num, rng, T, method = 'kmeans++')
    return centroid
    
def ramdon_data(data, k ,rng ):
    result = []
#    np.random.seed(seed)
    for i in range(0,k):
        pick = data[rng.randint(0, len(data))]
#        for l in pick:
            #if not mykmeans.check_exist(l, result):
        result.append(pick)
    return result