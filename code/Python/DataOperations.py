import pandas
from sklearn import preprocessing
import numpy as np

#Class Description: To read and scale data
class DataOperations:

    #Description: To read file 
    #Parameters: filePath = path of file to read, sep = separator used, indexCol = column of file to be index
    #Returns: pandas DataFrame
    def readData(self, filePath, sep, indexCol):
        return pandas.read_csv(filePath, sep=sep, header=0, index_col=indexCol)
       
    #Description: To zero scale values
    #Parameters: values = values to be zero scaled
    #Returns: scaledValues = scaled values  
    def zeroScale(self, values): 
        scaler = preprocessing.MinMaxScaler()
        scaledValues = scaler.fit_transform(values)
        return scaledValues
    
    #Description: Unit norm scale data
    #Parameters: values = values to be unit norm scaled
    #Returns: The scaled values
    def unitNorm(self, values):
        return preprocessing.normalize(values)
    
    #Description: Median scale normalise values
    #Parameters: values = the values to normalise
    #Returns: values = the normalised values 
    def medianScaleNormalisation(self, values):
        medianx = values.median(axis='columns')
        madx = values.mad(axis='columns')
        values = values.sub(medianx, axis='index') 
        values = values.multiply(1/madx, axis='index')
        return values
    
    #Description: Robust scale values
    #Parameters: values = the values to scale
    #Returns: scaled = the scaled values
    def robustScaler(self, values):
        values = values.transpose()
        scaled = preprocessing.RobustScaler().fit_transform(values)
        scaled = scaled.transpose()
        return scaled
    
    #Description: Collapse array into string separated by 'v', for comparison of groups
    #Parameters: array = array containing one piece of information for groups e.g. size
    #Returns: arrayToStr  = single string containing contents of array      
    def collapseArray(self, array):
        array = np.asarray(array)
        arrayToStr = array.astype(str)
        arrayToStr = ' v '.join(arrayToStr)
        return arrayToStr