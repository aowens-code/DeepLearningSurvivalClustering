import numpy as np
from scipy.spatial import distance
     
#Class Description: To determine centroids and group assignments
class CentroidOperations:
            
    #Description: Recalculate centroids based on data and groups
    #Parameters: data = data to calculate centroids using, groups = groups to recalculate centroids with, k = number of centroids required
    #Returns: centroids = the centroids 
    def recalculateCentroids(data, groups, k):
        centroids = np.zeros((k,data.shape[1]))
        for x in range(k):
            centroid = np.mean(data[groups == x], axis=0)  
            centroids[x] = centroid
        return centroids   
       
    #Description: Get group assignments by assigning samples to groups to which their bottleneck has the shorted squared euclidean distance
    #Parameters: data = data from which to assign groups, centroids = centroids for groups, k = number of groups
    #Returns: groupAssignments = the group assignments
    def getGroupAssignments(data, centroids, k):
        distances = np.zeros((k,data.shape[0])) 
        for x in range(k):         
            centroid =  np.repeat([centroids[x]],data.shape[0],axis=0)
            d = np.sum(np.square(np.subtract(data, centroid)), axis=1)
            distances[x] = d
        groupAssignments = np.argmin(distances, axis=0) 
        
        return groupAssignments
             
    #Description: Determine the furthest pair of centroids using euclidean distance  
    #Parameters: data = data from which to pick centroids
    #Returns: centroids = selected centroids
    def getFurthestCentroidPair(data): 
        distances = distance.cdist(data, data, 'euclidean')
        largestDistance = np.unravel_index(distances.argmax(), distances.shape)
        centroids = np.zeros((2,data.shape[1]))

        for x in np.arange(2):
            centroid = data[largestDistance[x]]
            centroids[x] = centroid

        return centroids
    