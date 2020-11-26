
import tensorflow as tf
import numpy as np
from sklearn import metrics
import pandas
from sklearn.cluster import KMeans

#Class Description: Contains custom callback implementation for on_epoch_end method, a callback used with survival based losses. 
#Calculates silhouette score per epoch
class SurvivalCallback(tf.keras.callbacks.Callback):

    def __init__(self, training, k):
        self.training = training 
        self.k = k
        self.silhouette = np.array([])
        self.groupAssignments = []
  
    #Description: This callback is called after each epoch completes. It updates the group assignments
    #Parameters: epoch = the epoch just complete
    def on_epoch_end(self, epoch, logs={}):  

        encoder = tf.keras.Model(inputs=self.model.input,outputs=self.model.get_layer('bottleneck').output)   
        #Get bottleneck 
        numpyBottleneck = np.array(encoder.predict(self.training)) 
        
        #Update group assignments
        newAssignments = []
        kmeans = KMeans(n_clusters=2, algorithm="full",init="k-means++")
        newAssignments = kmeans.fit_predict(numpyBottleneck)
                
        bottleneckSilhouetteScore= metrics.silhouette_score(pandas.DataFrame(numpyBottleneck), newAssignments, metric='euclidean')      
        self.silhouette = np.append(self.silhouette, bottleneckSilhouetteScore)     
        self.groupAssignments = newAssignments
        