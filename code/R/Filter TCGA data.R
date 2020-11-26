setUpEnvironment <- function(tcgaAssemblerPath){
	setwd(tcgaAssemblerPath)
	source(paste(tcgaAssemblerPath, "/Module_A.R", sep=""))
	source(paste(tcgaAssemblerPath, "/Module_B.R", sep=""))
}

createDirectory <- function(dirPath){
	dir.create(dirPath, showWarnings = FALSE, recursive=TRUE)
}

downloadClinicalData <- function(clinicalDataFolder, cancerType){
		biospecimenData <- DownloadBiospecimenClinicalData(cancerType, saveFolderName = clinicalDataFolder, outputFileName = "biospecimenData")
}

prepareClinicalDataframe <- function(clinicalDf){
	#Remove top rows, headings are duplicated
	clinicalDf <- clinicalDf[-c(1:2),]
	return(clinicalDf)
}

filterClinicalDataframe <- function(clinicalDf, diagnosis, grades, patients){
  
  clinicalDf <- subset(clinicalDf, !(is.na(as.numeric(clinicalDf[,'death_days_to'])) & is.na(as.numeric(clinicalDf[,'last_contact_days_to']))))
  
  #Select specified patients
  clinicalDf <- subset(clinicalDf, clinicalDf[,'bcr_patient_barcode'] %in% patients)
  
  #Remove patients with survival/last contact days under zero
  clinicalDf <- clinicalDf[!(clinicalDf[,'last_contact_days_to'] != '[Not Available]' & clinicalDf[,'last_contact_days_to'] < 0),]
  clinicalDf <- clinicalDf[!(clinicalDf[,'death_days_to'] != '[Not Applicable]' & clinicalDf[,'death_days_to'] < 0),]
  
  #Subset by diagnosis
  if(diagnosis[1] != 'All'){
    clinicalDf <- subset(clinicalDf, clinicalDf[,'histologic_diagnosis'] %in% diagnosis)
  }
  
  #Subset by grade
  if(grades[1] != 'All'){
    clinicalDf <- subset(clinicalDf, clinicalDf[,'tumor_grade'] %in% grades)	
  }
  
  return(clinicalDf)
}

getBarcodesFromClinicalData <- function(clinicalData){
	return(clinicalData$bcr_patient_barcode)
}

downloadGeneExpressionData <- function(downloadedExpressionOutputFolder, tissueTypes, cancerType, patients){
		DownloadRNASeqData(cancerType = cancerType, assayPlatform="gene.normalized_RNAseq", saveFolderName= downloadedExpressionOutputFolder, tissueType=tissueTypes, inputPatientIDs =patients)	
}

processGeneExpression <- function(downloadedExpressionOutputFolder, processedExpressionOutputFolder){
	downloadedGeneExpressionFilePath = list.files(downloadedExpressionOutputFolder, full.names=TRUE)[1]
	  processed <-ProcessRNASeqData(inputFilePath= downloadedGeneExpressionFilePath, outputFileName="processedExpressionData", outputFileFolder = processedExpressionOutputFolder , verType="RNASeqV2", dataType="geneExp")	
	  return(processed)
}

downloadMiRNAData <- function(downloadedMiRNAOutputFolder, tissueTypes, cancerType, patients){
    DownloadmiRNASeqData(cancerType = cancerType, saveFolderName= downloadedMiRNAOutputFolder, tissueType=tissueTypes, assayPlatform="mir_HiSeq.hg19.mirbase20", inputPatientIDs =patients)	
}

processMiRNA <- function(downloadedMiRNAOutputFolder, processedMiRNAOutputFolder){
  downloadedMiRNAFilePath = list.files(downloadedMiRNAOutputFolder, full.names=TRUE)[1]
    processed <- ProcessmiRNASeqData(inputFilePath= downloadedMiRNAFilePath, outputFileName="processedMiRNA", outputFileFolder = processedMiRNAOutputFolder )	
    return(processed)
}

downloadMethylationData <- function(downloadedMethylationOutputFolder, tissueTypes, cancerType, patients){
    methylation <- DownloadMethylationData(cancerType, assayPlatform = "methylation_450", tissueType = tissueTypes, saveFolderName = downloadedMethylationOutputFolder, outputFileName = "DownloadedMethylation", inputPatientIDs =patients )
    return(methylation)
}

processMethylation <- function(downloadedMethylationFilePath, processedMethylationOutputFolder, outputFileName, tcgaAssemblerPath){
    methylationData <- ProcessMethylation450Data(inputFilePath = downloadedMethylationFilePath, outputFileName= outputFileName, outputFileFolder=processedMethylationOutputFolder, fileSource = "TCGA-Assembler")
    averageValues <- CalculateSingleValueMethylationData(input = methylationData, regionOption = "TSS1500", DHSOption = "Both", outputFileName = outputFileName, outputFileFolder = processedMethylationOutputFolder, chipAnnotationFile = paste(tcgaAssemblerPath, "/SupportingFiles/MethylationChipAnnotation.rda", sep=''))
    return(averageValues)
}

filterOmicsFeaturesAndSamples <- function(filteredOmics){
	#Remove features with 20% or more of their values zero or missing 
	filteredOmics <- filteredOmics[apply(filteredOmics, 1, function(row){sum(is.na(row)) +  sum(row == 0, na.rm=TRUE) <= ncol(filteredOmics)/5 } ),]
	#Remove samples who have 20% or more of their features missing
	filteredOmics <- filteredOmics[,apply(filteredOmics, 2, function(col){ sum(is.na(col)) +  sum(col == 0, na.rm=TRUE)  <= nrow(filteredOmics)/5} )]
	return(filteredOmics)
}

imputeOmics <-function(filteredOmics){
	
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("impute")
library(impute)

imputedOmics <- impute.knn(as.matrix(filteredOmics))$data
return(imputedOmics)
	
}

#SETUP

#Set TCGA assembler path
tcgaAssemblerPath <- ""
#Set cancer and tissue type
tissueTypes <- c("TP")
cancerType <- 'LIHC'
grades <- c("All")
diagnosis <- c('Hepatocellular Carcinoma')
#Set folder to save data
baseFolderAll <- ''

#SET UP
setUpEnvironment(tcgaAssemblerPath)

#CREATE SUBFOLDERS IN BASE FOLDER
clinicalDataFolder <- paste(baseFolderAll, '/ClinicalData', sep="")
createDirectory(clinicalDataFolder)

downloadedExpressionOutputFolder <- paste(baseFolderAll,'/DownloadedGeneExpression', sep="")
createDirectory(downloadedExpressionOutputFolder)

processedExpressionOutputFolder <- paste(baseFolderAll,'/ProcessedGeneExpression', sep="")
createDirectory(processedExpressionOutputFolder)

downloadedMiRNAOutputFolder <- paste(baseFolderAll,'/DownloadedMiRNA', sep="")
createDirectory(downloadedMiRNAOutputFolder)

processedMiRNAOutputFolder <- paste(baseFolderAll,'/ProcessedMiRNA', sep="")
createDirectory(processedMiRNAOutputFolder)

downloadedMethylationOutputFolder <- paste(baseFolderAll,'/DownloadedMethylation', sep="")
createDirectory(downloadedMethylationOutputFolder)

processedMethylationOutputFolder <- paste(baseFolderAll,'/ProcessedMethylation', sep="")
createDirectory(processedMethylationOutputFolder)

processedMultiOmicsFolder <- paste(baseFolderAll,'/ProcessedMultiOmics', sep="")
createDirectory(processedMultiOmicsFolder)

#DOWNLOAD CLINICAL 
downloadClinicalData(clinicalDataFolder, cancerType)
clinicalDataPath = paste(clinicalDataFolder, '/biospecimenData__nationwidechildrens.org_clinical_patient_', tolower(cancerType), '.txt',sep="")

#READ CLINICAL 
clinicalData <- read.delim(clinicalDataPath, sep='\t',stringsAsFactors=FALSE, header=TRUE)
clinicalData <- prepareClinicalDataframe(clinicalData)

#GET PATIENT BARCODES
barcodes <- getBarcodesFromClinicalData(clinicalData)
barcodeChunks <- split(barcodes, ceiling(seq_along(barcodes)/76))

#DOWNLOAD OMICS DATA
downloadGeneExpressionData(downloadedExpressionOutputFolder, tissueTypes, cancerType, barcodes)
processedExpression <- processGeneExpression(downloadedExpressionOutputFolder, processedExpressionOutputFolder)

downloadMiRNAData(downloadedMiRNAOutputFolder, tissueTypes, cancerType, barcodes)
processedMiRNA <- processMiRNA(downloadedMiRNAOutputFolder, processedMiRNAOutputFolder)

batchOne <- downloadMethylationData(downloadedMethylationOutputFolder, tissueTypes, cancerType, barcodeChunks[[1]])
batchOne <- paste(downloadedMethylationOutputFolder, "/DownloadedMethylation__LIHC__methylation_450__TP__20200414002618.txt",sep="")
processedMethylationOne <- processMethylation(batchOne, processedMethylationOutputFolder, "processedMethylationBatchOne", tcgaAssemblerPath)

batchTwo <- downloadMethylationData(downloadedMethylationOutputFolder, tissueTypes, cancerType, barcodeChunks[[2]])
batchTwo <- paste(downloadedMethylationOutputFolder, "/DownloadedMethylation__LIHC__methylation_450__TP__20200414005247.txt",sep="")
processedMethylationTwo <- processMethylation(batchTwo, processedMethylationOutputFolder, "processedMethylationBatchTwo", tcgaAssemblerPath)

batchThree <- downloadMethylationData(downloadedMethylationOutputFolder, tissueTypes, cancerType, barcodeChunks[[3]])
batchThree <- paste(downloadedMethylationOutputFolder, "/DownloadedMethylation__LIHC__methylation_450__TP__20200414011115.txt",sep="")
processedMethylationThree <- processMethylation(batchThree, processedMethylationOutputFolder, "processedMethylationBatchThree", tcgaAssemblerPath)

batchFour <- downloadMethylationData(downloadedMethylationOutputFolder, tissueTypes, cancerType, barcodeChunks[[4]])
batchFour <- paste(downloadedMethylationOutputFolder, "/DownloadedMethylation__LIHC__methylation_450__TP__20200414013024.txt",sep="")
processedMethylationFour <- processMethylation(batchFour, processedMethylationOutputFolder, "processedMethylationBatchFour", tcgaAssemblerPath)

batchFive <- downloadMethylationData(downloadedMethylationOutputFolder, tissueTypes, cancerType, barcodeChunks[[5]])
batchFive<- paste(downloadedMethylationOutputFolder, "/DownloadedMethylation__LIHC__methylation_450__TP__20200414015540.txt",sep="")
processedMethylationFive <- processMethylation(batchFive, processedMethylationOutputFolder, "processedMethylationBatchFive", tcgaAssemblerPath)

combineMethylation <- cbind(processedMethylationOne$Des, processedMethylationOne$Data)
combineMethylation <- cbind(combineMethylation, processedMethylationTwo$Data)
combineMethylation <- cbind(combineMethylation, processedMethylationThree$Data)
combineMethylation <- cbind(combineMethylation, processedMethylationFour$Data)
combineMethylation <- cbind(combineMethylation, processedMethylationFive$Data)
write.table(combineMethylation, file = paste(processedMethylationOutputFolder, '/combinedMethylation.txt', sep=""), row.names = FALSE,col.names=TRUE ,sep = "\t")

exp <- read.delim(paste(processedExpressionOutputFolder, '/processedExpressionData.txt',sep=""), row.names=2, header = TRUE, stringsAsFactors=FALSE,  sep="\t")
exp <- exp[,-c(1)]

miRNA <- read.delim(paste(processedMiRNAOutputFolder, '/processedMiRNA__ReadCount.txt',sep=""), header = TRUE, row.names = 1, stringsAsFactors=FALSE,  sep="\t")

meth <- read.delim(paste(processedMethylationOutputFolder, '/combinedMethylation.txt', sep=""), header = TRUE, row.names = 1, stringsAsFactors=FALSE,  sep="\t")
meth <- meth[,-c(1)]

colnames(meth) <- gsub("[.]","-",substring(colnames(meth), 1, 15))
colnames(exp) <- gsub("[.]","-",substring(colnames(exp), 1, 15))
colnames(miRNA) <- gsub("[.]","-", substring(colnames(miRNA), 1, 15))
commonSamples <- intersect(intersect(colnames(meth) , colnames(exp)), colnames(miRNA))
commonPatients <- substring(commonSamples, 1, 12)

filteredClinical <- filterClinicalDataframe(clinicalData, diagnosis, grades, commonPatients)

#UPDATE COMMON PATIENTS AFTER CLINICAL FILTERING
commonPatients <- filteredClinical[,'bcr_patient_barcode']

#FILTER OMICS
filteredExpression <- exp
colnames(filteredExpression) <- substring(colnames(filteredExpression), 1, 12)
filteredExpression <- filteredExpression[,commonPatients]

filteredMethylation <- meth
colnames(filteredMethylation) <- substring(colnames(filteredMethylation), 1, 12)
filteredMethylation <- filteredMethylation[,commonPatients]

filteredMiRNA <- miRNA
colnames(filteredMiRNA) <- substring(colnames(filteredMiRNA), 1, 12)
filteredMiRNA <- filteredMiRNA[,commonPatients]

#PREPROCESS OMICS
filteredExpression <- filterOmicsFeaturesAndSamples(filteredExpression)
filteredMethylation <- filterOmicsFeaturesAndSamples(filteredMethylation)
filteredMiRNA <- filterOmicsFeaturesAndSamples(filteredMiRNA)

#IMPUTE
imputedExpression <- imputeOmics(filteredExpression)  
imputedMethylation <-  imputeOmics(filteredMethylation)  
imputedMiRNA <- imputeOmics(filteredMiRNA)  

#Write preprocessing files
write.csv(imputedExpression, file = paste(processedMultiOmicsFolder, '/expression.csv', sep=""), row.names = TRUE )
write.csv(imputedMethylation, file = paste(processedMultiOmicsFolder, '/methylation.csv', sep=""), row.names = TRUE )
write.csv(imputedMiRNA, file = paste(processedMultiOmicsFolder, '/mirna.csv', sep=""), row.names = TRUE )
write.csv(filteredClinical, file = paste(processedMultiOmicsFolder, '/filteredClinical.csv', sep=""), row.names = FALSE )
