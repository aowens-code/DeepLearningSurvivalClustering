import pandas
import numpy as np

#Class Description: To prepare TCGA data for analysis
class PrepareDataTCGA:
    
    def __init__(self, rootPath, clinical):
        self.rootPath = rootPath
        self.clinical = clinical
      
    #Description: Prepare clinical data
    #Returns: clinicalDf = clinical information   
    def getClinical(self): 
        clinical = self.clinical
        clinical['death_days_to'] = clinical['death_days_to'].replace('[Not Applicable]', np.nan)
        
        tcgaId = clinical['bcr_patient_barcode']
        os = pandas.to_numeric(clinical['death_days_to'].fillna(clinical['last_contact_days_to']))
        status = np.where((clinical.vital_status == 'Dead'),1,0)
        grade = clinical['tumor_grade']
        data = {'TCGA_ID': tcgaId, 'Censor':status, 'OS':os, 'Grade': grade}
        clinicalDf = pandas.DataFrame(data, columns=['TCGA_ID', 'Censor','OS', 'Grade'])
        return clinicalDf
             
    #Description: Stack the multi-omics data into a single matrix
    #Parameters: patientIDs = patient identifiers, methylationData = methylation data, miRNAData = miRNA data, rnaData = RNA data
    #Returns: omicsMatrix = single matrix containing all three omics types
    def getMultiOmicsMatrix(self, patientIDs, methylationData, miRNAData, rnaData): 
                   
        #If all column headings identical
        if(np.logical_and( (methylationData.columns==rnaData.columns).all(), (rnaData.columns==miRNAData.columns).all() )):
            #Concatenate dataframes
            omicsMatrix = pandas.concat([methylationData, rnaData, miRNAData], axis=0)
            omicsMatrix = omicsMatrix[patientIDs] 
            return omicsMatrix
    
    