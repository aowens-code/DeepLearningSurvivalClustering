import lifelines 
import numpy as np
import pandas

#Class Description: To generate survival statistics and plots
class SurvivalAnalysis:
    def __init__(self, clinical, dataSourceID):
         self.clinical = clinical
         self.dataSourceID= dataSourceID
         
    #Description: Plot Kaplan-Meier plot for specified groups
    #Parameters: names = Sample ids, filePath = path to save plot to, groups = group labels, chosenAnalysis = analysis to use in plot title, k = number of clusters
    def kaplanMeierPlotForGroups(self, names, filePath, groups, chosenAnalysis, k):
        kmf = lifelines.KaplanMeierFitter() 
        colours = ['m', 'c','b','y','r']
        results = self.logRankPairwise(self.clinical[self.dataSourceID], groups)
        pValue = np.mean(results.p_value)
        
        groupClinical = self.clinical[self.clinical[self.dataSourceID].isin(names[0])]
        groupTotal = groupClinical.shape[0]
        kmf.fit(groupClinical['OS'], groupClinical['Censor'],label='Group '+ str(1) + ' ('+str(groupTotal)+')')
        ax = kmf.plot(ci_show=False, show_censors=True, color=colours[0])
        
        for x in np.arange(1,k):
             groupClinical = self.clinical[self.clinical[self.dataSourceID].isin(names[x])]
             groupTotal = groupClinical.shape[0]
             kmf.fit(groupClinical['OS'], groupClinical['Censor'],label='Group '+ str(x+1) +' ('+str(groupTotal)+')')
             plot = kmf.plot(ax=ax, ci_show=False, show_censors=True, color=colours[x])
            
        plot.set_xlabel('Days')
        plot.set_ylabel('Survival Probability')
        plot.set_title(chosenAnalysis + ': Log-rank p-value = ' + str(format(pValue, '.3g')))
        plot.get_figure().savefig(filePath)
    
    #Description: Split data by median value and create two cohorts
    #Parameters: data = omics data, feature = feature values are to be split on 
    #Returns: the group labels 
    def getMedianGroups(self, data, feature):
        featureValues = data.loc[feature,:]
        if len(featureValues.shape) > 1:
            featureValues = pandas.DataFrame.mean(featureValues, axis=0)  
        median = np.median(featureValues)
        groups = featureValues <= median
        groups = groups.astype(int)
        return groups
    
    #Description: Get log-rank p-value of median value split of feature
    #Parameters: data = omics data, feature = feature values are to be split on 
    #Returns: the log-rank p-value
    def getMedianLogRank(self, data, feature):
        groups = self.getMedianGroups(data, feature)
        results = self.logRankPairwise(data.columns, groups)
        pValue = results.p_value[0]
        return pValue
    
    #Description: Plots Kaplan-Meier for median split on feature
    #Parameters: feature = feature values are to be split on ,symbol = symbol for feature to be used in plot title, data = omics data, filePath = path to save plot to
    def kaplanMeierPlotForFeature(self, feature, symbol, data, filePath):        
        featureValues = data.loc[feature,:]
        
        #Average values in there are duplicates
        if len(featureValues.shape) > 1:
            featureValues = pandas.DataFrame.mean(featureValues, axis=0)   
            
        groups = self.getMedianGroups(data, feature)
        pValue = self.getMedianLogRank(data, feature)
        lower = featureValues[groups == 1]
        upper = featureValues[groups == 0]
        lowerClinical = self.clinical[self.clinical[self.dataSourceID].isin(lower.index.values)]
        upperClinical = self.clinical[self.clinical[self.dataSourceID].isin(upper.index.values)]
        kmf = lifelines.KaplanMeierFitter() 
        lowerTotal = lowerClinical.shape[0]
        upperTotal = upperClinical.shape[0]
        kmf.fit(lowerClinical['OS'], lowerClinical['Censor'],label='Low Expression (' + str(lowerTotal)+')')
        ax = kmf.plot(ci_show=False, show_censors=True)
        kmf.fit(upperClinical['OS'], upperClinical['Censor'],label='High Expression (' + str(upperTotal)+')')
        plot = kmf.plot(ax=ax, ci_show=False, show_censors=True)
        plot.set_xlabel('Days')
        plot.set_ylabel('Survival Probability')
        plot.set_title(symbol + ' Log-rank p-value = ' +str(format(pValue, '.3g')))
        plot.get_figure().savefig(filePath)
    
    #Description: Performs log-rank test for specified labels 
    #Parameters: names = sample IDs, labels = group labels     
    #Returns: the lifelines StatisticalResult object
    def logRankPairwise(self, names, labels):     
        clinical = self.clinical[self.clinical[self.dataSourceID].isin(names)]
        clinical.reset_index(drop=True, inplace=True)
        durations = clinical['OS']
        events = clinical['Censor']
        results = lifelines.statistics.pairwise_logrank_test(durations, labels, events)
        return results   
 