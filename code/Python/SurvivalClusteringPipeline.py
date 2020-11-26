import pandas
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sb
from collections import Counter
from matplotlib.patches import Patch

from SurvivalClustering.SurvivalAnalysis import SurvivalAnalysis
from SurvivalClustering.Cluster import Cluster
from SurvivalClustering.DataOperations import DataOperations
from SurvivalClustering.CentroidOperations import CentroidOperations
from SurvivalClustering.OutputFormatting import OutputFormatting
from SurvivalClustering.PrepareDataTCGA import PrepareDataTCGA
from SurvivalClustering.FeatureOperations import FeatureOperations
from SurvivalClustering.CustomLosses import CustomLosses
from SurvivalClustering.Autoencoders import Autoencoders
from SurvivalClustering.ClusterCallback import ClusterCallback
from SurvivalClustering.SurvivalCallback import SurvivalCallback
      
#Class Description: Perform analysis    
class SurvivalClusteringPipeline():
        
    def __init__(self, rootPath, outputFolderPath) :
        
        self.outputFolderPath = outputFolderPath + "/"
        self.dataOperations = DataOperations()
        self.output = OutputFormatting(rootPath, self.outputFolderPath)
        self.rootPath = rootPath   
        self.dataSourceID = "TCGA_ID"
        self.rnaFileName = "expression.csv"
        self.methylationFileName = "methylation.csv"
        self.miRNAFileName = "mirna.csv"
        clinicalPath = rootPath + "/" + "filteredClinical.csv"
        clinical = self.dataOperations.readData(clinicalPath, ",", None) 
        self.prepareData = PrepareDataTCGA(rootPath, clinical) 
    
    #Description: Plot training loss
    #Parameters: history = model history, chosenLoss = chosen loss, noEpochs = number of epochs model run for, outputFileName = file name for output, epochLabels = labels to use for x (epoch) axis
    def plotLoss(self, history, chosenLoss, noEpochs, outputFileName, epochLabels):
         plt.close('all')
         plt.plot(history.history['loss'])
         plotTitle = 'Training Loss for ' + OutputFormatting.getChosenAnalysisFullTitle(chosenLoss) + ' (' + str(noEpochs) + ' epochs)'
         plt.title(plotTitle)
         plt.ylabel('loss')
         plt.xlabel('epochs')
         epochIndex = epochLabels-1
         plt.xticks(epochIndex, epochLabels)
         plt.legend(['train'], loc='upper left')
         self.output.writePlot(plt, outputFileName+"_Loss")
  
    #Description: Plot silhouette score 
    #Parameters: silhouette = array of silhouette scores, chosenLoss = chosen loss, noEpochs = number of epochs model run for, outputFileName= file name for output, epochLabels = labels to use for x (epoch) axis
    def plotSilhouette(self, silhouette, chosenLoss, noEpochs, outputFileName, epochLabels):
         plt.close('all')
         plt.plot(silhouette)
         plotTitle = 'Silhouette scores for ' + OutputFormatting.getChosenAnalysisFullTitle(chosenLoss) + ' (' + str(noEpochs) + ' epochs)'
         plt.title(plotTitle)
         plt.ylabel('score')
         plt.xlabel('epochs') 
         epochIndex = epochLabels-1
         plt.xticks(epochIndex, epochLabels)
         self.output.writePlot(plt, outputFileName+"_Silhouette")
     
    #Description: Get bottleneck for custom loss function     
    #Parameters: training = training data, bottleneckDim = bottleneck dimension, epochs = number of epochs, k = number of clusters, 
    #chosenAnalysis = chosen analysis, clinical = clinical information, hiddenDim = dimension of hidden layer, activation = activation function, optimizer = optimizer to be used
    #Returns: bottleneckFeatures = bottleneck features, history = model history, assignments = group assignments, silhouette = silhouette score per epoch
    def getBottleneckFeaturesCustom(self, training, bottleneckDim, epochs, k, chosenAnalysis, clinical, hiddenDim, activation, optimizer):
         numSamples = training.shape[0] 
         originalDim = training.shape[1]
         
         if chosenAnalysis == 'survival':     
             survivalCallback = SurvivalCallback(training, k)
             censor = np.float32(clinical['Censor'])
             model = Autoencoders.getAutoencoderSurvival(originalDim, bottleneckDim, activation, hiddenDim)
             model.compile(loss=[CustomLosses.reconstruction, CustomLosses.survivalLoss],loss_weights=[0.25,0.75], optimizer=optimizer) 
             history = model.fit(training, [training, censor], batch_size=numSamples, epochs=epochs, shuffle=False,  callbacks=[survivalCallback]) 
             assignments = survivalCallback.groupAssignments
             silhouette  = survivalCallback.silhouette
             
         elif chosenAnalysis == 'cluster_close_far' or chosenAnalysis == 'cluster_close_far_survival':
           
             model = Autoencoders.getAutoencoder(originalDim, bottleneckDim, activation, hiddenDim)
             model.compile(loss=CustomLosses.reconstruction, optimizer=optimizer)
             model.fit(training, training, batch_size=numSamples, epochs=1, shuffle=False)
             encoder = tf.keras.Model(inputs=model.input,outputs=model.get_layer('bottleneck').output)
             
             #Get prediction for bottleneck layer         
             initialBottleneckPrediction = np.array(encoder.predict(training))
                         
             #Seed centroids from bottleneck layer
             centroids = np.zeros((k,bottleneckDim))
             centroids = CentroidOperations.getFurthestCentroidPair(initialBottleneckPrediction)
    
             #Calculate group assignments and determine closest and furthest centroids
             groupAssignments = CentroidOperations.getGroupAssignments(initialBottleneckPrediction, centroids, k)         
             
             #Create backend variables to hold both the centroids and the group assignments   
             groupAssignments = tf.keras.backend.variable(np.array(groupAssignments))
             groupCentroids = tf.keras.backend.variable(centroids)
                  
             #Set as class variables in  CustomLosses class              
             CustomLosses.groupAssignments = groupAssignments
             CustomLosses.groupCentroids = groupCentroids         
             custom_callback = ClusterCallback(training, numSamples, bottleneckDim, k)  
             
             if chosenAnalysis == 'cluster_close_far':  
                 model = Autoencoders.getAutoencoderBottleneck(originalDim, bottleneckDim, activation, hiddenDim) 
                 model.compile(loss=[CustomLosses.reconstruction, CustomLosses.clusterLossCloseFar],loss_weights=[0.25,0.75], optimizer=optimizer) 
                 history = model.fit(training, [training, training], batch_size=numSamples, epochs=epochs, callbacks=[custom_callback] , shuffle=False)
                 assignments = CustomLosses.groupAssignments
                 assignments = assignments.numpy()
                 silhouette  = custom_callback.silhouette
             elif chosenAnalysis == 'cluster_close_far_survival':
                 censor = np.float32(clinical['Censor'])
                 model = Autoencoders.getAutoencoderBottleneckSurvival(originalDim, bottleneckDim, activation, hiddenDim)
                 model.compile(loss=[CustomLosses.reconstruction, CustomLosses.clusterLossCloseFar, CustomLosses.survivalLoss],loss_weights=[0.25, 0.25, 0.50], optimizer=optimizer)   
                 history = model.fit(training, [training, training, censor], batch_size=numSamples, epochs=epochs, shuffle=False, callbacks=[custom_callback])
                 assignments = CustomLosses.groupAssignments
                 assignments = assignments.numpy()
                 silhouette  = custom_callback.silhouette
   
         bottleneck = tf.keras.Model(inputs=model.input,outputs=model.get_layer('bottleneck').output)
         
         #Make bottleneck prediction using trained model    
         prediction = bottleneck.predict(training)
         bottleneckFeatures = pandas.DataFrame(np.array(prediction))
         return bottleneckFeatures, history, assignments, silhouette
         
    #Description: Get methylation data
    #Returns: methylationData = methylation data 
    def getMethylation(self):
        methylationPath = self.rootPath + "/" + self.methylationFileName
        methylationData = self.dataOperations.readData(methylationPath, ",", 0)
        return methylationData
    
    #Description: Get miRNA data
    #Returns: miRNAData = miRNA data 
    def getMiRNA(self):
        miRNAPath =  self.rootPath + "/" + self.miRNAFileName
        miRNAData = self.dataOperations.readData(miRNAPath, ",", 0)
        return miRNAData
    
    #Description: Get RNA data
    #Returns: rnaData = RNA data 
    def getRNAData(self):
        rnaPath = self.rootPath + "/" + self.rnaFileName
        rnaData = self.dataOperations.readData(rnaPath, ",", 0)
        return rnaData    
     
    #Description: Get multi-omics data
    #Parameters: identifiers = identifiers to get data for 
    #Returns: methylation = methylation data, micro = miRNA data, rna = RNA data
    def getOriginalFeatures(self, identifiers):
         methylation = self.getMethylation().loc[:,identifiers]
         micro = self.getMiRNA().loc[:,identifiers]
         rna = self.getRNAData().loc[:,identifiers]
         return methylation, micro, rna
     
    #Description: Generate clusters from bottleneck features and compile analysis
    #Parameters: clinical = clinical data, k = number of clusters, outputFileName = output file name, bottleneckFeatures = bottleneck features, clusters = cluster information, runNo = current run number
    #Returns: clusterAnalysis = array containing results of cluster analysis 
    def analyseClusters(self, clinical , k, outputFileName, bottleneckFeatures, clusters, runNo):             
        survivalAnalysis = SurvivalAnalysis(clinical, self.dataSourceID)
        results = survivalAnalysis.logRankPairwise(clinical[self.dataSourceID], clusters.labels)
        pValue = results.p_value[0]
        pValueRounded = format(pValue, '.2f')
        bottleneckSilhouetteScore= metrics.silhouette_score(bottleneckFeatures, clusters.labels, metric='euclidean')
        bottleneckSilhouetteScoreRound= format(bottleneckSilhouetteScore, '.2f')
        clusterSizes = self.dataOperations.collapseArray(clusters.getClustersSize())
        clusterSurvivalTimes = self.dataOperations.collapseArray(clusters.getMeanClustersSurvival(clinical))
        clusterSurvivalTimesRounded = self.dataOperations.collapseArray(np.round(clusters.getMeanClustersSurvival(clinical), 2))
        noBottleneckFeatures = bottleneckFeatures.shape[1]     
        clusterAnalysis = [str(runNo), clusterSizes, bottleneckSilhouetteScore, bottleneckSilhouetteScoreRound, clusterSurvivalTimes, clusterSurvivalTimesRounded, pValue, pValueRounded, noBottleneckFeatures]   
        return clusterAnalysis
      
    #Description: Get scaled omics data
    #Parameters: clinical = clinical information
    #Returns: scaledRNA = scaled RNA data, scaledMethylation = scaled methylation data, scaledMiRNA = scaled miRNA data
    def getScaledOmics(self, clinical):
        methylation, micro, rna = self.getOriginalFeatures(clinical[self.dataSourceID])
        scaledRNA = pandas.DataFrame(self.dataOperations.robustScaler(self.dataOperations.medianScaleNormalisation(rna)), columns=rna.columns, index=rna.index.values)
        scaledMethylation = pandas.DataFrame(self.dataOperations.robustScaler(self.dataOperations.medianScaleNormalisation(methylation)), columns=methylation.columns, index=methylation.index.values)
        scaledMiRNA= pandas.DataFrame(self.dataOperations.unitNorm(self.dataOperations.medianScaleNormalisation(micro)), columns=micro.columns, index=micro.index.values)
        return scaledRNA, scaledMethylation, scaledMiRNA
        
    #Description: Get ANOVA p-value for omics features
    #Parameters: clusters = cluster information, clinical = clinical information, k = number of clusters
    #Returns: rnaAnova = Anova p-values for RNA, methylationAnova = Anova p-values for methylation, miRNAAnova= Anova p-values for miRNA
    def determineANOVAFeatureRankingInClusters(self, clusters, clinical, k):
        scaledRNA, scaledMethylation, scaledMiRNA = self.getScaledOmics(clinical)  
        scaledRNA.to_csv(self.outputFolderPath + '/scaledRNA.csv')
        scaledMethylation.to_csv(self.outputFolderPath + '/scaledMeth.csv')
        scaledMiRNA.to_csv(self.outputFolderPath + '/scaledmiRNA.csv')
        methylationAnova = FeatureOperations.getAnovaOriginalFeatures(scaledMethylation, clusters.labels, k)
        miRNAAnova = FeatureOperations.getAnovaOriginalFeatures(scaledMiRNA, clusters.labels, k)
        rnaAnova = FeatureOperations.getAnovaOriginalFeatures(scaledRNA, clusters.labels, k)    
        return rnaAnova, methylationAnova, miRNAAnova
      
    #Description: Get common features among runs
    #Parameters: features = omics features, overlap = number of runs to overlap , omicsData = omics data, clinical = clinical data
    #Returns: countOverlapFeatures = common features among specified number of runs 
    def commonFeatures(self, features, overlap, omicsData, clinical):
        logRank= []
        survivalAnalysis = SurvivalAnalysis(clinical, self.dataSourceID)
        counts = Counter(features)
        countDf = pandas.DataFrame.from_dict(counts, orient='index').reset_index()
        countDf.columns = ['Feature', 'Count']
        countOverlapFeatures = countDf[countDf['Count'] >= overlap]
        for feature in countOverlapFeatures['Feature']:  
            p = survivalAnalysis.getMedianLogRank(omicsData, feature)  
            logRank.append(p)
        countOverlapFeatures['Median Log-rank'] = logRank
        return countOverlapFeatures
                  
    #Description: Perform custom analysis 
    #Parameters: bottleneckDim = bottleneck dimension, hiddenDim = hidden layer dimension, activation = activation function, chosenAnalysis = chosen analysis, 
    #epochs = number of epochs, optimizer = optimiser, noRuns = number of runs to run analysis for, portionFeatures = portion of runs, significanceThreshold = ANOVA p-value threshold, 
    #runsOverlap = array for run overlap
    def runAnalysis(self, bottleneckDim, hiddenDim, activation, chosenAnalysis, epochs, optimizer, noRuns, portionFeatures, significanceThreshold, runsOverlap):
        #Optimal number of clusters chosen as 2
        k = 2
        topRNAs = []
        topMethylation =[]
        topMiRNA = []
        clusterLabels = []
        clusterAnalyses = []
     
        for x in np.arange(1, noRuns+1):
            outputFileName = chosenAnalysis + '_run_' + str(x) + '_' + str(epochs) + '_' + str(k) 
            clinical = self.prepareData.getClinical()  
            
            if chosenAnalysis == 'survival' or chosenAnalysis == 'cluster_close_far_survival':
                clinical = clinical.sort_values(by='OS', ascending=False)
               
            identifiers = clinical[self.dataSourceID]
            array = self.prepareData.getMultiOmicsMatrix(identifiers, self.getMethylation(), self.getMiRNA(), self.getRNAData()).transpose()
            scaledData = pandas.DataFrame(self.dataOperations.zeroScale(array), index=array.index.values, columns=array.columns)
            
            bottleneckFeatures, history, assignments, silhouetteScores = self.getBottleneckFeaturesCustom(scaledData, bottleneckDim, epochs, k, chosenAnalysis, clinical, hiddenDim, activation, optimizer)  
            bottleneckFeatures.index =   identifiers
            self.output.writeCsv(outputFileName + '_BottleneckFeatures', bottleneckFeatures, True)
            epochLabels = np.arange(10, epochs+1, step=10)
            self.plotLoss(history, chosenAnalysis, epochs, outputFileName, epochLabels)           
            self.plotSilhouette(silhouetteScores, chosenAnalysis, epochs, outputFileName, epochLabels)
            
            clusters = Cluster(bottleneckFeatures, clinical[self.dataSourceID], k, self.dataSourceID)
            clusters.labels = assignments
            
            clusterAnalysis = self.analyseClusters(clinical, k, outputFileName, bottleneckFeatures, clusters, x)
            clusterAnalyses.append(clusterAnalysis) 

            rnaAnova, methylationAnova, microAnova= self.determineANOVAFeatureRankingInClusters(clusters, clinical, k) 

            topRNAs.append(rnaAnova['p-value'])
            topMethylation.append(methylationAnova['p-value'])
            topMiRNA.append(microAnova['p-value'])
            clusterLabels.append(clusters.labels)
            survivalAnalysis = SurvivalAnalysis(clinical, self.dataSourceID)
            survivalAnalysis.kaplanMeierPlotForGroups(clusters.getClusterNames(), self.outputFolderPath + outputFileName+ '.png', clusters.labels, OutputFormatting.getChosenAnalysisFullTitle(chosenAnalysis), k)
        
        self.output.writeSurvivalFile(self.dataSourceID, identifiers, clinical, chosenAnalysis+'_AllRunsSurvival', clusterLabels)
       
        allRnaAnova = self.output.getANOVAFeatures(topRNAs, rnaAnova['Genes'])
        allMethylationAnova = self.output.getANOVAFeatures(topMethylation, methylationAnova['Genes'])
        allMiRNAAnova = self.output.getANOVAFeatures(topMiRNA, microAnova['Genes'])
             
        self.output.writeCsv(chosenAnalysis+'_AnovaRna', allRnaAnova, True)
        self.output.writeCsv(chosenAnalysis+'_AnovaMethylation',allMethylationAnova, True)
        self.output.writeCsv( chosenAnalysis+'_AnovaMiRNA',allMiRNAAnova, True)
        self.output.writeClusterMetrics(pandas.DataFrame(clusterAnalyses), chosenAnalysis+"_ClusterAnalysis")
          
        for runOverlap in runsOverlap:
            self.getOverlappingFeatures(allRnaAnova, portionFeatures, significanceThreshold, runOverlap, chosenAnalysis+'OverlapRNA', 'rna')
            self.getOverlappingFeatures(allMethylationAnova, portionFeatures, significanceThreshold, runOverlap, chosenAnalysis+'OverlapMethylation', 'methylation')
            self.getOverlappingFeatures(allMiRNAAnova, portionFeatures, significanceThreshold, runOverlap, chosenAnalysis+'OverlapMiRNA', 'miRNA')
        
    #Description: Generate heatmap for data and labels 
    #Parameters: omicsData = omics data, labels = group labels, clusterColours = colours to use for clusters, outputFileName = output file name, featureCol = column to use for feature
    def generateHeatmap(self, omicsData, labels, clusterColours, outputFileName, featureCol): 
        clinical = self.prepareData.getClinical()
        data = {'IDs': omicsData.columns, 'Labels': labels}    
        df = pandas.DataFrame(data=data) 
        df.set_index('IDs', inplace=True)
        df = df.loc[clinical[self.dataSourceID]]
        df['Grade'] = clinical['Grade'].values
        df = df.sort_values(['Labels', 'Grade'])
        clusterColColours = np.array([])
        for label in df['Labels']:
            clusterColColours = np.append(clusterColColours, clusterColours[label])

        colourDict = {'G1':'cyan', 'G2':'violet', 'G3': 'indigo', 'G4': 'green', '[Not Available]': 'lightgray'}

        gradeColours = df['Grade'].map(colourDict)      
        
        patch = [Patch(facecolor=colourDict[name]) for name in colourDict]
        
        columnColours = [clusterColColours, gradeColours]
        
        omicsDataForHeatmap = omicsData[df.index.values] 
        sb.clustermap(omicsDataForHeatmap, cmap='vlag', col_cluster=False,col_colors=columnColours, z_score=0, vmin=-3, vmax=3, xticklabels=False, yticklabels=featureCol, figsize=(20, 20))
        plt.legend(patch, colourDict,bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right',prop={'size': 25})
        plt.savefig(self.outputFolderPath + outputFileName + '_Heatmap.png', dpi=400)
   
    #Description: Generate heatmap 
    #Parameters: patientIds = patient identifiers, clusterLabels = cluster labels, featureIds = feature ids, outputFileName = output file name,featureLabels = feature labels, featureCol = feature column to use
    def heatmapForRna(self, patientIds, clusterLabels, featureIds, outputFileName, featureLabels, featureCol):
        methylation, micro, rna = self.getOriginalFeatures(patientIds)
        rnaForHeatmap = rna.loc[featureIds,:]
        if featureLabels is not None:
            rnaForHeatmap.index = featureLabels
        else:
            rnaForHeatmap.index = featureIds 
        
        self.generateHeatmap(rnaForHeatmap, clusterLabels, ['r', 'b'], outputFileName, featureCol)
    
    #Description: To determine overlapping features
    #Parameters: anovaPath = file path to csv file containing anova p-values, portionFeatures = consider this number of features with the lowest ANOVA p-values, 
    #significanceThreshold = significance threshold to use for ANOVA p-value, runsOverlap=portion of runs to examine for overlap, fileName  = name of file, dataType = type of omics data   
    def getOverlappingFeaturesFromPath(self, anovaPath, portionFeatures, significanceThreshold, runsOverlap, fileName, dataType):
        anovaFeatures = pandas.read_csv(anovaPath,index_col=0)
        for runOverlap in runsOverlap:
            self.getOverlappingFeatures(anovaFeatures, portionFeatures, significanceThreshold, runOverlap, fileName, dataType)
       
    #Description: To determine overlapping features
    #Parameters: anovaFeatures = anova p-values, portionFeatures = consider this number of features with the lowest ANOVA p-values, 
    #significanceThreshold = significance threshold to use for ANOVA p-value, runOverlap=portion of runs to examine for overlap, fileName  = name of file, dataType = type of omics data   
    def getOverlappingFeatures(self, anovaFeatures, portionFeatures, significanceThreshold, runOverlap, fileName, dataType):
        clinical = self.prepareData.getClinical()
        selectedGenes = np.array([])
        
        for column in anovaFeatures:
            sortedValues = anovaFeatures[column].sort_values()  
               
            selectedSortedValues = sortedValues[0:round(len(sortedValues) * portionFeatures)]
                  
            if significanceThreshold is not None:
                selectedSortedValues = selectedSortedValues[selectedSortedValues < significanceThreshold]
            
            selectedGenes = np.append(selectedGenes, selectedSortedValues.index.values)
            
        scaledRNA, scaledMethylation, scaledMiRNA = self.getScaledOmics(clinical)
        
        if dataType == 'rna':
            common = self.commonFeatures(selectedGenes, runOverlap, scaledRNA, clinical)
        elif dataType == 'methylation':
            common = self.commonFeatures(selectedGenes, runOverlap, scaledMethylation, clinical)
        elif dataType == 'miRNA':
            common = self.commonFeatures(selectedGenes, runOverlap, scaledMiRNA, clinical)
        
        self.output.writeCsv(fileName + '_' + str(runOverlap) + '_runs', common, False)
        
    #Description: Generate heatmap 
    #Parameters: patientIds = patient identifiers, clusterLabels = cluster labels, featureIds = feature ids, outputFileName = output file name,featureLabels = feature labels, featureCol = feature column to use
    def heatmapForMethylation(self, patientIds, clusterLabels, featureIds, outputFileName, featureLabels, featureCol):
        methylation, micro, rna = self.getOriginalFeatures(patientIds)
        methylationForHeatmap = methylation.loc[featureIds,:]
        if featureLabels is not None:
            methylationForHeatmap.index = featureLabels
        else:
            methylationForHeatmap.index = featureIds 
        
        self.generateHeatmap(methylationForHeatmap, clusterLabels, ['r', 'b'], outputFileName, featureCol)
        
    #Description: Creates initial bottleneck to be used for determining optimal number of k
    #Parameters: bottleDim = bottleneck dimension, hiddenDim = hidden dimension, activation = activation function
    def createInitialBottleneck(self, bottleDim, hiddenDim, activation):        
        clinical = self.prepareData.getClinical() 
        identifiers = clinical[self.dataSourceID]      
        array = self.prepareData.getMultiOmicsMatrix(identifiers, self.getMethylation(), self.getMiRNA(), self.getRNAData()).transpose() 
        training = pandas.DataFrame(self.dataOperations.unitNorm(array), index=array.index.values, columns=array.columns) 
        originalDim = training.shape[1]
        model = Autoencoders.getAutoencoder(originalDim, bottleDim, activation, hiddenDim)
        model.compile(loss='mse', optimizer='adam')  
        history = model.fit(training, training, batch_size=training.shape[0], epochs=10) 
        bottleneck = tf.keras.Model(inputs=model.input,outputs=model.get_layer('bottleneck').output)             
        prediction = bottleneck.predict(training)
        bottleneckFeatures = pandas.DataFrame(np.array(prediction)) 
        bottleneckFeatures.index = identifiers
        self.output.writeCsv('initialBottleneck', bottleneckFeatures, True)    

                