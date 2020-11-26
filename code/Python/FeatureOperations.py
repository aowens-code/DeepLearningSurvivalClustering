import scipy.stats as stats
import pandas

#Class Description: To perform analysis on features
class FeatureOperations:
    
    #Description: Get anova p value of features between groups
    #Parameters: data = omics data, groups = cluster labels, k = number of groups
    #Returns: pvalueDf = p-values for each feature
    def getAnovaOriginalFeatures(data, groups, k):
          pValues = []
          for x in range(0,data.shape[0]): 
              groupGene= []
              for y in range(0,k):
                 
                  item = data.iloc[x,groups==y]
                  groupGene.append(item)
              F, p = stats.f_oneway(*groupGene)
              pValues.append(p) 
           
          pvalueResults = {'Genes': data.index.values , 'p-value': pValues}
          pvalueDf = pandas.DataFrame(pvalueResults, columns=['Genes', 'p-value'])     
          return pvalueDf
