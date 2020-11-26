import sys 
#Download python filesin repository and insert their path here to import them
sys.path.append("")
from SurvivalClustering.SurvivalClusteringPipeline import SurvivalClusteringPipeline
import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

#rootFolder should contain path to processed data to be used for analysis
#outputFolder should contain path where output files should be placed
rootFolder = ''
outputFolder = ''    

bottleneckDim = 100
hiddenDim = 1000
activation = 'sigmoid'
optimizer = 'adam'

noRuns = 10
portionFeatures = 0.1
significanceThreshold = 0.005
runsOverlap = [6, 8, 10]

#Create initial bottleneck to cluster in order to determine optimum number of k
survivalClusteringPipeline = SurvivalClusteringPipeline(rootFolder,outputFolder)
survivalClusteringPipeline.createInitialBottleneck(bottleneckDim, hiddenDim, activation)

#Preliminary cluster analysis 
#Read csv contain bottleneck being used for preliminary analysis
bottleneck = pandas.read_csv("", index_col=0)
kmeans = KMeans(n_clusters=2, algorithm="full",init="k-means++")
labels = kmeans.fit_predict(bottleneck)    
bottleneckSilhouetteScore= metrics.silhouette_score(bottleneck, labels, metric='euclidean')

kmeans = KMeans(n_clusters=3, algorithm="full",init="k-means++")
labels = kmeans.fit_predict(bottleneck)    
bottleneckSilhouetteScore= metrics.silhouette_score(bottleneck, labels, metric='euclidean')

kmeans = KMeans(n_clusters=4, algorithm="full",init="k-means++")
labels = kmeans.fit_predict(bottleneck)    
bottleneckSilhouetteScore= metrics.silhouette_score(bottleneck, labels, metric='euclidean')

kmeans = KMeans(n_clusters=5, algorithm="full",init="k-means++")
labels = kmeans.fit_predict(bottleneck)    
bottleneckSilhouetteScore= metrics.silhouette_score(bottleneck, labels, metric='euclidean')

#Optimum number of k chosen as 2

#Perform analysis using LRC, LRS and LRSC on custom autoencoder
#LRC  = 'cluster_close_far'
#LRSC = 'cluster_close_far_survival'
#LRS = 'survival'
method = 'cluster_close_far'
survivalClusteringPipeline = SurvivalClusteringPipeline(rootFolder,outputFolder)
survivalClusteringPipeline.runAnalysis(bottleneckDim, hiddenDim, activation, method, 40, optimizer, noRuns, portionFeatures, significanceThreshold, runsOverlap)

method = 'cluster_close_far_survival'
survivalClusteringPipeline = SurvivalClusteringPipeline(rootFolder,outputFolder)
survivalClusteringPipeline.runAnalysis(bottleneckDim, hiddenDim, activation, method, 40, optimizer, noRuns, portionFeatures, significanceThreshold, runsOverlap)

method = 'survival'
survivalClusteringPipeline = SurvivalClusteringPipeline(rootFolder,outputFolder)
survivalClusteringPipeline.runAnalysis(bottleneckDim, hiddenDim, activation, method, 40, optimizer, noRuns, portionFeatures, significanceThreshold, runsOverlap)

#Generate heatmaps          
#Read csv containing relevant survival file with cluster labels
survivalFile = pandas.read_csv("")
patientIds = survivalFile['ID']
#Select labels from run to be analysed e.g. Run 1 labels
labels = np.array(survivalFile['Run 1 labels']).astype(int)

#RNA Heatmap 
#Read csv containing entrez ids to create heatmap with 
significantFeatures = pandas.read_csv("").iloc[:,0]
#Read csv containing gene symbols for entrez ids 
significantFeaturesSymbols = pandas.read_csv("").iloc[:,1]
#Create heatmap        
survivalClusteringPipeline = SurvivalClusteringPipeline(rootFolder,outputFolder)
survivalClusteringPipeline.heatmapForRna(patientIds, labels, significantFeatures, 'heatmap_name', significantFeaturesSymbols, 1) 

#Methylation
#Read csv containing methylation symbols to create heatmap with
significantFeatures = pandas.read_csv("").iloc[:,0]
#Create heatmap  
survivalClusteringPipeline.heatmapForMethylation(patientIds, labels, significantFeatures, 'heatmap2_name', None, False) 
