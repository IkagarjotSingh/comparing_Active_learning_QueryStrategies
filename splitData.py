import os
import warnings
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import logs
import argparse

#Sample Command to execute this file
#python3 splitData.py -i "/data/preprocessed/firefox_data.csv" -o "/data/processed/firefox"

def get_args():
    '''
    Validates and retuns Run time/ Command Line arguments.
    '''
    parser = argparse.ArgumentParser(description="This script takes requirement combination file as input and return train, test, validation and to be predicted datasets. This specific activity allows to take care of the randomness in generating initial train and test sets for model prediction and comparison acitivties.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
     
    parser.add_argument("--input","-i",type=str, required = True, help= "path to processed requirement combination file")
    parser.add_argument("--fraction","-f",type=str,default = "0.4", required = False,help= "the fraction of requirement combinations to be marked as manually labelled, inorder to have the initial training set for model (balanced - undersampling) - Default value 0.4")
    parser.add_argument("--output","-o",type=str, required = True,help= "directory path to save train, test, validation and tobepredicted requirement combinations files")
    #Oversampling
    args = parser.parse_args()
    return args

def getData(fPath,frac):

    '''
    Fetches data from the file path provided.
    Randomly select the data and marks 'frac' of it as manually annotated after data balancing (under sampling).
    
    Parameters : 
    fPath (str) : File path from which data is to be extracted
    frac (str) : Fraction of requirement combinations that need to marked as Manually Annotated (for initial training set)

    Returns :
    df_balancedComb (DataFrame) : Balanced Data marked 'M' in annotation status 
    df_remainingComb (DataFrame) :  Remaining Data 
    '''
    print("\nGenerating dataframe from the input file.")
    df_data = pd.read_csv(fPath,',',encoding="utf-8",error_bad_lines=False)
    df_data['AnnotationStatus'] = ""
    
    print("\nMarking "+str(100*float(frac))+"% of data as Manually Annotated (After data balancing).")
    df_balancedComb, df_remainingComb = balanceDataSet(df_data,"Label",frac)
    
    df_balancedComb['AnnotationStatus'] = "M" #Mark the sampled values as Annotation 'M' - these combinations will be the inital training dataset.
    #Combine sampled combinations (Labelled as manually annotations) and the remaining requirement combinations 
    #df_finalComb = shuffle(pd.concat([df_balancedComb,df_remainingComb]))

    return df_balancedComb,df_remainingComb

def balanceDataSet(df_rqmts, colName ,frac):
    '''
    Performs data balancing by undersampling.
        1. Does a value count for each class label and extacts the minimum value.
        2. Selects fraction * minimum value (Undersampling) of the data points from each class.

    Parameters : 
    df_rqmts (DataFrame) : Dataframe containing requirement combinations and corresponding labels.
    colName (str) : Name of the column on which value counts/ data balancing is to be performed
    frac (str) : Fraction of requirement combinations that need to separated.

    Returns 
        df_sampledCombinations (DataFrame) : Sampled Balanced Dataset 
        df_rqmts (DataFrame) :  Remaining Dataset
    '''

    print("\nPerforming Data Balancing :-")
    print ("\nOriginal Size of Dataset: "+str(len(df_rqmts)))
    print ("\nValue Count of column '"+colName+"' : \n"+str(df_rqmts[colName].value_counts()))
    df_rqmts[colName] = df_rqmts[colName].astype('int')
    stats = df_rqmts[colName].value_counts()  #Returns a series of number of different types of TargetLabels (values) available with their count.
    
    count = int(stats.min()*float(frac))
    print ("\nSampled Combinations for each class : "+str(count) + " ("+str(frac)+" of the total combinations)") 
        
    df_sampledCombinations = pd.DataFrame(columns=df_rqmts.columns)
    
    for key in stats.keys():
        #Sample out some values for df_data Set for each label 0,1,2,3
        df_sample = df_rqmts[df_rqmts[colName]==key].sample(count)
        #df_sample['AnnotationStatus'] = "M" #Mark the sampled values as Annotation 'M' - these combinations will be the inital training dataset.
        df_rqmts = df_rqmts[~df_rqmts.isin(df_sample)].dropna()  #Remove Sampled Values from original data set.
        df_sampledCombinations = pd.concat([df_sampledCombinations,df_sample],axis=0)   #Add sampled values into the Test Set

    print ("\nSize of Sampled Combinations : "+str(len(df_sampledCombinations)))
    print ("\nSize of Remaining Combinations : "+str(len(df_rqmts)))
    
    return df_sampledCombinations,df_rqmts    

def main():
    #Ignore Future warnings if any occur. 
    warnings.simplefilter(action='ignore', category=FutureWarning)  
    
    #To make sure all the columns are visible in the logs.
    pd.set_option('display.max_columns', 500)   
    pd.set_option('display.width', 1000)

    #initialize directory which contains all the data and which will contain logs and outputs
    currentFileDir = os.getcwd()
    
    args=get_args()

    ifileName = currentFileDir+args.input
    fraction = args.fraction
    odirName = currentFileDir+args.output
    
    #logFilePath,OFilePath = logs.createLogs(currentFileDir+"/Logs",args,"Data Splitting")   #Creates the log file, default value is os.getcwd()+"/static/data/logs/" ; user still can provide his own logPath if needed.
    cols = ['comboId','req1Id','req1','req_1','req2Id','req2','req_2','Label','AnnotationStatus']
    print ("\n"+"-"*150)
    print("Fetching data from the input file and Marking "+str(float(fraction)*100)+"% of the combinations as Manually Annotated. Setting 'Annotation Status' flas as 'M'")
    print ("-"*150)
    df_ManuallyAnnotatedSet,df_toBeAnnotatedSet = getData(ifileName,fraction) 
    df_ManuallyAnnotatedSet = df_ManuallyAnnotatedSet[cols]
    df_toBeAnnotatedSet = df_toBeAnnotatedSet[cols]
    
    print ("\n"+"-"*100)
    print("\nSplitting the Manually annotated data into validation set and initial traning/test set")
    print ("-"*100)
    df_validationSet,df_labelledSet = balanceDataSet(df_ManuallyAnnotatedSet,"Label",0.2)
    
    print ("\n"+"-"*100)
    print("\nSplitting the Manually annotated data into validation set and initial traning/test set")
    print ("-"*100)
    
    df_trainingSet,df_testSet = balanceDataSet(df_labelledSet,"Label",0.8)
    
    print ("\nSaving the datasets after splitting at : "+odirName)
    df_ManuallyAnnotatedSet.to_csv(odirName+"/ManuallyAnnotated.csv",index=False)
    df_toBeAnnotatedSet.to_csv(odirName+"/ToBeAnnotated.csv",index=False)
    df_validationSet.to_csv(odirName+"/ValidationSet.csv",index=False)
    df_trainingSet.to_csv(odirName+"/TrainingSet.csv",index=False)
    df_testSet.to_csv(odirName+"/TestSet.csv",index=False)

if __name__ == '__main__':
    main()