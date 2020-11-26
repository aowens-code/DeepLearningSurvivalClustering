import pandas
import numpy as np

#Class Description: Class to compile and write result outputs 
class OutputFormatting:
   
    def __init__(self, rootPath, outputFolderPath):
        self.outputFolderPath = outputFolderPath
      
    #Description: Write file detailing cluster metrics
    #Parameters: df = DataFrame containing cluster metrics, outputFileName = file name for output     
    def writeClusterMetrics(self, df, outputFileName):
        columns = ['Run No',                         
                  'Clusters Size',
                  'Silhouette Score Bottleneck',          
                  'Silhouette Score Bottleneck (2dp)',           
                  'Clusters Survival',
                  'Clusters Survival (2dp)',
                  'Log rank p-value',
                  'Log rank p-value (2dp)',
                  'No Bottleneck Features']
        df.columns = columns
        self.writeCsv(outputFileName, df, False)
     
    #Description: Write file containing survival information for clusters
    #Parameters: dataSourceID = data source, identifiers = patient identifiers, clinical = clinical information, outputFileName = file name for output, clusterLabels = labels for samples                        
    def writeSurvivalFile(self, dataSourceID, identifiers, clinical, outputFileName, clusterLabels):
        runHeadings = []
        for x in np.arange(1, len(clusterLabels)+1):
            runHeadings.append("Run " + str(x) +" labels")
        survivalDf = pandas.DataFrame({dataSourceID: identifiers,
                                     'Time': clinical['OS'],
                                     'Event': clinical['Censor'],
                                     'Grade': clinical['Grade']
                                     })
        labelsDf = pandas.DataFrame(clusterLabels).transpose()  
        labelsDf.columns = runHeadings
        columnNames= ['ID','Time','Event','Grade']+runHeadings
        survivalDf.reset_index(drop=True, inplace=True)
        labelsDf.reset_index(drop=True, inplace=True)
        overallDf = pandas.concat([survivalDf, labelsDf], axis=1, ignore_index=True) 
        overallDf.columns = columnNames
        self.writeCsv(outputFileName, overallDf, False)
           
    #Description: Write file containing ANOVA values 
    #Parameters: omicsPValues = ANOVA values for omics features, omicsFeatures = omics features 
    #Returns: ANOVA feature dataframe    
    def getANOVAFeatures(self, omicsPValues, omicsFeatures):
        df = pandas.DataFrame(omicsPValues)
        df.columns = omicsFeatures
        df = df.transpose()
        headings = []

        for x in np.arange(len(df.columns)):
            headings.append('ANOVA p-value Run ' + str(x+1))
            
        df.columns = headings    
        return df
    
    #Description: Save a plot
    #Parameters: plot = the plot to save, outputFileName = file name for plot               
    def writePlot(self, plot, outputFileName):      
        plotPath =  self.outputFolderPath + '/' + outputFileName + '.png'
        plot.savefig(plotPath)
    
    #Description: Save a DataFrame as a csv file
    #Parameters:  outputFileName = file name for output, df = DataFrame to write, writeIndex = whether or not to include index               
    def writeCsv(self, outputFileName, df, writeIndex):
         wholeFilePath = self.outputFolderPath + '/' + outputFileName + '.csv'
         df.to_csv(wholeFilePath, index=writeIndex)
      
    #Description: Get a title based on chosen loss
    #Parameters: chosenLoss = the chosen loss
    #Returns: the full title for the chosen loss
    def getChosenAnalysisFullTitle(chosenLoss):
        if chosenLoss == 'cluster_close_far':
            return 'Cluster Based Loss'
        elif chosenLoss == 'survival':
            return 'Survival Based Loss'
        elif chosenLoss == 'cluster_close_far_survival':
            return 'Survival Cluster Based Loss'
        
       