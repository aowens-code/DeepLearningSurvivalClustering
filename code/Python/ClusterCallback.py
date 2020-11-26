from SurvivalClustering.CentroidOperations import CentroidOperations
import tensorflow as tf
import numpy as np
from SurvivalClustering.CustomLosses import CustomLosses
from sklearn import metrics
import pandas

#Class Description: Contains custom callback implementation for on_epoch_end method, a callback used with clustering based losses. 
#It updates the class variables in the CustomLosses class before the next epoch
class ClusterCallback(tf.keras.callbacks.Callback):

    def __init__(self, training, numSamples,bottleneckDim, k):
        self.training = training 
        self.numSamples = numSamples
        self.k = k
        self.bottleneckDim = bottleneckDim
        self.silhouette = np.array([])
     
    #This callback is called after each epoch completes. It updates the centroids, group assignments and closest/furthest centroids
        
    #Description: This callback is called after each epoch completes. It updates the centroids and group assignments 
    #Parameters: epoch = the epoch just complete
    def on_epoch_end(self, epoch, logs={}):  
        #Get sample labels 
        groups = np.array(tf.keras.backend.get_value(CustomLosses.groupAssignments))  
        
        #Get encoder portion of model currently being trained
        encoder = tf.keras.Model(inputs=self.model.input,outputs=self.model.get_layer('bottleneck').output)   
        #Get bottleneck 
        numpyBottleneck = np.array(encoder.predict(self.training)) 
        
        #Update centroids using new bottleneck
        centroids = CentroidOperations.recalculateCentroids(numpyBottleneck, groups, self.k) 

        #Update group assignments using new centroids and bottleneck
        newAssignments = CentroidOperations.getGroupAssignments(numpyBottleneck, centroids, self.k)  

        bottleneckSilhouetteScore= metrics.silhouette_score(pandas.DataFrame(numpyBottleneck), newAssignments, metric='euclidean')      
        self.silhouette = np.append(self.silhouette, bottleneckSilhouetteScore)     

        #Update variables which hold furthest centroids, closest centroids and the group assignments for each sample
        tf.keras.backend.set_value(CustomLosses.groupAssignments, tf.convert_to_tensor(newAssignments)) 
        tf.keras.backend.set_value(CustomLosses.groupCentroids, centroids) 
        
        