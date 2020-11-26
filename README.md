# Novel Deep Learning-based Solution for Identification of Prognostic Subgroups in Hepatocellular Carcinoma

Deep learning based solution for creating prognostic subgroups in The Cancer Genome Atlas (TCGA) Hepatocellular carcinoma (HCC) samples. There are both Python and R parts to the solution.  

## Dependencies 

### Python

* Tensorflow 2.1.0
* Pandas 0.25.2
* Scikit-learn 0.22.2.post1
* Lifelines 0.22.9
* Numpy 1.17.3
* Seaborn 0.10.1
* Matplotlib 3.1.1
* Scipy 1.4.1

### R

* Impute 1.56.0
* Survival 3.2.3
* TCGA-Assembler 2.0

## Downloading Data

In this repository see the file 'Download and Pre-process TCGA data.R' in code/R for the code to download and pre-process TCGA HCC data using TCGA-Assembler 2.0. TCGA-Assembler 2.0 should be downloaded first and is available here https://github.com/compgenome365/TCGA-Assembler-2

## Analysing Data

The RUN.py file in code/Python shows how to set up resources and perform analysis

## Survival Analysis

For post-hoc survival analysis of deep learning features see UnivariateBottleneckAnalysis.R in code/R
