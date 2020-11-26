import numpy as np
from sklearn.cluster import KMeans

#Class Description: To cluster data
class Cluster:
    
    def __init__(self, omicsData, names, k, dataSourceID):
        
        self.omicsData = omicsData
        self.names = names
        self.k = k  
        self.labels = []
        self.dataSourceID = dataSourceID
    
    #Description: Perform k-means clustering on omics data 
    def kMeansCluster(self):
        kmeans = KMeans(n_clusters=self.k, algorithm="full",init="k-means++")
        kmeans.fit(self.omicsData)    
        self.labels = kmeans.predict(self.omicsData) 
      
    #Description: Get the sample ids of each cluster
    #Returns: the sample ids for each cluster
    def getClusterNames(self):
        groupsNames = []
        for i in range(0,self.k): 
            group = self.names[self.labels == i]
            groupsNames.append(group) 
        return groupsNames
 
    #Description: Get the average survival time for each cluster  
    #Parameters: clinical = the clinical data    
    #Returns: the mean survival for each cluster
    def getMeanClustersSurvival(self, clinical):
        groupsSurvival = []
        for i in range(0,self.k): 
            group = self.names[self.labels == i]
            groupOS = clinical[clinical[self.dataSourceID].isin(group)]['OS']  
            groupsSurvival.append(np.mean(groupOS)) 
        return groupsSurvival
      
    #Description: Get the sizes of each cluster    
    #Returns: the sizes of each cluster           
    def getClustersSize(self):
        groupsSize = []
        for i in range(0,self.k): 
            group = self.names[self.labels == i]
            groupsSize.append(group.size)        
        return groupsSize
    
        