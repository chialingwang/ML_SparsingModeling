import sys,os
import numpy as np
from sklearn.utils.extmath import squared_norm

def Kmeans_multi(data, k , rng , T = 100  ,  opt = 'True' , method = 'kmeans' ):
    outLoop = 0
    clusters = []
    min_dist = sys.maxsize
    centroids = []
    centroid = []
    lable = []
    dist = []
    print("This is Kmeans_Multi")
    while(outLoop < T):
        if(opt == 'True'):
            if(method == 'kmeans++'): 
                clusters, centroid , lable = kmeansopt(data, k  , rng, 50 , method = 'kmeans++' )
            else:
                clusters, centroid , lable = kmeansopt(data, k  , rng, 50   )
        else :
            if(method == 'kmeans++'): 
                clusters, centroid , lable = kmeans(data, k  , rng, 50 , method = 'kmeans++'  )
            else:
                clusters, centroid , lable = kmeans(data, k  , rng, 50 )
        new_min = sum_dist(clusters,centroid)
        dist.append(new_min)
        centroids.append(centroid)
        if( new_min < min_dist):
            min_dist = new_min
            clusters = clusters
            centroid = centroid
            
            lable = lable 
        outLoop += 1

    return clusters, centroids , lable , dist

def sum_dist(clusters , centroids):
    sum = 0
    for i in range(0,len(clusters)):
        for point in clusters[i]:
            sum += np.linalg.norm(np.array(point)-np.array(centroids[i])) 
    return sum

def kmeans(data, k   ,rng, T = 50  , method = 'kmeans' ):
    centroids = []
#    centroid = []
    lable = []
    if(method == 'kmeans++'): 
        centroids = optimize_centroids(data, centroids , k  ,rng )
    else:
        centroids = ramdon_centroids(data, centroids , k  ,rng)
        #centroids = [[0,0], [0,0.01], [0.01,0]]
    print("inital centroids")
    print(centroids)
    old_centroids = [[] for i in range(k)]
    #    result_dict = {}
    Iteration = 0
    clusters = [[] for i in range(k)]
    #    while(Iteration < T and not compare(old_centroids , centroids)):
    while(Iteration < T ):
        clusters = [[] for i in range(k)]
        clusters,lable= euclidean(data, centroids, clusters)
    #        print(" The %d times cluster" % Iteration)
    #    print(clusters)
            # recalculate centriods from exist cluster
        index = 0
        old_centroids = list(centroids);
#        centroid.append(centroids)
        for cluster in clusters:
#            old_centroids[index] = centroids[index];
            centroids[index] = np.mean(cluster, axis = 0).tolist()
            index += 1
        Iteration += 1    # End of innerLoop
    
#    for num in range(0,len(clusters)):
#        for ld in clusters[num]:
#            result_dict[str(ld)] = num
#    print(centroids)    
    return clusters, centroids, lable

def kmeansopt(data, k   ,rng, T = 50  , method = 'kmeans' , tol = 1e-4 ):
    centroids = []
    lable = []
    
    if(method == 'kmeans++'): 
        centroids = optimize_centroids(data, centroids , k  ,rng )
    else:
        centroids = ramdon_centroids(data, centroids , k  ,rng)
#    print("inital centroids")
#    print(centroids)
    old_centroids = []
    #    result_dict = {}
    Iteration = 0
    clusters = [[] for i in range(k)]
    #    while(Iteration < T and not compare(old_centroids , centroids)):
    while(Iteration < T ):
        clusters = [[] for i in range(k)]
        clusters,lable= euclidean(data, centroids, clusters)
    #        print(" The %d times cluster" % Iteration)
    #    print(clusters)
            # recalculate centriods from exist cluster
        index = 0
        old_centroids = list(centroids);
#        print(Iteration)
        for cluster in clusters:
            centroids[index] = np.mean(cluster, axis = 0).tolist()
            index += 1
            
#    for num in range(0,len(clusters)):
#        for ld in clusters[num]:
#            result_dict[str(ld)] = num
#        print(centroids)
        centroids_matrix = np.matrix(centroids)
#        print(centroids_matrix)
#        print(old_centroids)
        old_centroids_matrix = np.matrix(old_centroids)
#        print(old_centroids_matrix)
        shift = squared_norm(old_centroids_matrix - centroids_matrix)
        
        if shift <= tol:
#            print("Already Coverage , break")
            break
        
        Iteration += 1    # End of innerLoop
    return clusters, centroids, lable


def euclidean(data, centroids, clusters):
    # find which centroids the x is closet to, and put x into the centroids location
    lable = []

    for x in data:
        min_dist = sys.maxsize
        index = 0
        for i in range(0,len(centroids)):

            if(np.linalg.norm(np.array(x)-np.array(centroids[i])) < min_dist ):
                       min_dist = np.linalg.norm(np.array(x)-np.array(centroids[i]))
                       index = i
        clusters[index].append(x)
        lable.append(index)
        
        
    return clusters, lable

def ramdon_centroids(data, centroids , k ,rng ):
#    np.random.seed(seed)
    i = 0
    while i < k:
        pick = data[rng.randint(0, len(data))]
#        print(pick)
        if not check_exist(pick, centroids):
            centroids.append(pick)
            i += 1
    return centroids

def check_exist(choose , centroids):
    for item in centroids:
#        print(choose)
#        print(item)
        if(np.array_equal(choose,item)):
            return True
    return False

def min_dist(data,centroids):
    min_dist_array = []
    for x in data:
        min_dist = sys.maxsize
        index = 0
        for i in range(0,len(centroids)):
            if(np.linalg.norm(np.array(x)-np.array(centroids[i])) < min_dist ):
                min_dist = np.linalg.norm(np.array(x)-np.array(centroids[i]))
        min_dist_array.append(min_dist)

    return min_dist_array



def optimize_centroids(data, centroids , k  ,rng):
#    print("This is Kmeans++")
#    np.random.Seed = seed
    pick = data[rng.randint(0, len(data))]
    centroids.append(pick)
    d = []
    i = 1
    while i < k:
        d = min_dist(data,centroids)
        sum = 0;
        for dist in d:
            sum += dist

        sum *= np.random.uniform(0, 1)
 
        for index in range(1,len(d)):
            sum -= d[index]
            if sum > 0:
                continue
            if not check_exist(data[index], centroids):
                centroids.append(data[index])
                i +=1  
            break
#    print("Now the Cnetroids for Kmeas++")        
#    print(centroids)       
    
    return centroids


def compare(old_centroids , centroids):
#    print(old_centroids)
#    print(centroids)
#    print(old_centroids == centroids);
    return (old_centroids == centroids)


def score(ori , new):
    err = 0
    for i in range(0,len(ori)):
        if(new[i] != ori[i]):
            err += 0
    return err/len(ori)