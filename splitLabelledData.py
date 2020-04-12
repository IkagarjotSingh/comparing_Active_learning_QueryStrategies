import os
import warnings
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import logs
import argparse


#Sample Command to execute this file
#python3 splitLabelledData.py -i "/data/preprocessed/firefox_data.csv" -o "/data/processed/firefox"

def get_args():
    '''
    Validates and retuns Run time/ Command Line arguments.
    '''
    parser = argparse.ArgumentParser(description="This script takes requirement combination file as input and return train, test, validation and to be predicted datasets. This specific activity allows to take care of the randomness in generating initial train and test sets for model prediction and comparison acitivties.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
     
    parser.add_argument("--input","-i",type=str, required = True, help= "path to processed requirement combination file")
    #parser.add_argument("--fraction","-f",type=str,default = "0.4", required = False,help= "the fraction of requirement combinations to be marked as manually labelled, inorder to have the initial training set for model (balanced - undersampling) - Default value 0.4")
    parser.add_argument("--output","-o",type=str, required = True,help= "directory path to save train, test, validation and tobepredicted requirement combinations files")
    #Oversampling
    args = parser.parse_args()
    return args


def splitDataSet(df_rqmts,frac,resamplingType):
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

    print("\nSplitting Data :-")
    print ("\nOriginal Size of Dataset: "+str(len(df_rqmts)))
    df_rqmts["Label"] = df_rqmts["Label"].astype('int')
    stats = df_rqmts["Label"].value_counts()  #Returns a series of number of different types of TargetLabels (values) available with their count.
    print ("Value Count : \n"+str(stats))

    #if resampleType == "under_sampling":
    #    count = int(stats.min()*float(frac))
    #print ("\nSampled Combinations for each class : "+str(count) + " ("+str(frac)+" of the total combinations)") 
        
    df_sampledCombinations = pd.DataFrame(columns=df_rqmts.columns)
    
    for key in stats.keys():
        #Sample out some values for df_data Set for each label 0,1,2,3
        if (resamplingType == "over_sampling"):
            sample_count = int(stats[key]*float(frac))  
            if (sample_count>500):  #Limiting Sample Count to 500
                sample_count = 500
        else:
            sample_count = sample_count = int(stats.min()*float(frac))

        df_sample = df_rqmts[df_rqmts["Label"]==key].sample(sample_count)
        print ("\nSampled "+str(len(df_sample))+" Combinations for class "+str(key) + " ("+str(frac)+" of the total combinations)") 
     
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
    odirName = currentFileDir+args.output
    
    print("\nGenerating dataframe from the input file.")
    df_data = pd.read_csv(ifileName,',',encoding="utf-8",error_bad_lines=False)
    df_data['AnnotationStatus'] = ""

    cols = ['comboId','req1Id','req1','req_1','req2Id','req2','req_2','Label','AnnotationStatus']
    
    print ("\nPreparing Data for Random Under Sampling")
    print ("\n"+"-"*150)
    print("Fetching data from the input file and Marking 40% of the balanced combinations as Manually Annotated. Setting 'Annotation Status' as 'M'")
    print ("-"*150)

    df_ManuallyAnnotatedSet,df_toBeAnnotatedSet = splitDataSet(df_data,0.4,"under_sampling") 
    df_ManuallyAnnotatedSet = df_ManuallyAnnotatedSet[cols]
    df_toBeAnnotatedSet = df_toBeAnnotatedSet[cols]
    df_ManuallyAnnotatedSet['AnnotationStatus'] = 'M'
    print ("df_ManuallyAnnnotatedSet \n"+str( df_ManuallyAnnotatedSet['Label'].value_counts()))

    print ("df_toBeAnnotatedSet \n"+str( df_toBeAnnotatedSet['Label'].value_counts()))

    print ("\n"+"-"*100)
    print("\nSplitting the Manually annotated data into validation set and initial traning/test set")
    print ("-"*100)
    df_validationSet,df_labelledSet = splitDataSet(df_ManuallyAnnotatedSet,0.2,"under_sampling")
    
    print ("df_validationSet \n"+str( df_validationSet['Label'].value_counts()))

    print ("df_labelledSet \n"+str( df_labelledSet['Label'].value_counts()))

    print ("\n"+"-"*100)
    print("\nSplitting the Manually annotated data into validation set and initial traning/test set")
    print ("-"*100)
    
    df_trainingSet,df_testSet = splitDataSet(df_labelledSet,0.8,"under_sampling")

    print ("df_trainingSet \n"+str( df_trainingSet['Label'].value_counts()))

    print ("df_testSet \n"+str( df_testSet['Label'].value_counts()))
    
    print ("\nSaving the datasets after splitting at : "+odirName+"/under_sampling")
    
    df_ManuallyAnnotatedSet.to_csv(odirName+"/under_sampling/ManuallyAnnotated.csv",index=False)
    df_toBeAnnotatedSet.to_csv(odirName+"/under_sampling/ToBeAnnotated.csv",index=False)
    df_validationSet.to_csv(odirName+"/under_sampling/ValidationSet.csv",index=False)
    df_trainingSet.to_csv(odirName+"/under_sampling/TrainingSet.csv",index=False)
    df_testSet.to_csv(odirName+"/under_sampling/TestSet.csv",index=False)
    
    
    ################################################################################################################################################################
    print ("\nPreparing Data for Random Over Sampling")
    print ("\n"+"-"*150)
    print("Marking 10% of the combinations for each class as Manually Annotated. Setting 'Annotation Status' as 'M'")
    print ("-"*150)

    df_ManuallyAnnotatedSet,df_toBeAnnotatedSet = splitDataSet(df_data,0.1 ,"over_sampling") 
    df_ManuallyAnnotatedSet = df_ManuallyAnnotatedSet[cols]
    df_toBeAnnotatedSet = df_toBeAnnotatedSet[cols]
    df_ManuallyAnnotatedSet['AnnotationStatus'] = 'M'
    
    print ("df_ManuallyAnnnotatedSet \n"+str( df_ManuallyAnnotatedSet['Label'].value_counts()))

    print ("df_toBeAnnotatedSet \n"+str( df_toBeAnnotatedSet['Label'].value_counts()))
    
    print ("\n"+"-"*100)
    print("\nSplitting the Manually annotated data into validation set and initial traning/test set")
    print ("-"*100)
    df_validationSet,df_labelledSet = splitDataSet(df_ManuallyAnnotatedSet,0.3,"over_sampling")
    
    print ("df_validationSet \n"+str( df_validationSet['Label'].value_counts()))

    print ("df_labelledSet \n"+str( df_labelledSet['Label'].value_counts()))
    

    df_trainingSet,df_testSet = splitDataSet(df_labelledSet,0.8,"over_sampling")

    print ("df_trainingSet \n"+str( df_trainingSet['Label'].value_counts()))

    print ("df_testSet \n"+str( df_testSet['Label'].value_counts()))

    print ("\nSaving the datasets after splitting at : "+odirName+"/over_sampling")
    
    df_ManuallyAnnotatedSet.to_csv(odirName+"/over_sampling/ManuallyAnnotated.csv",index=False)
    df_toBeAnnotatedSet.to_csv(odirName+"/over_sampling/ToBeAnnotated.csv",index=False)
    df_validationSet.to_csv(odirName+"/over_sampling/ValidationSet.csv",index=False)
    df_trainingSet.to_csv(odirName+"/over_sampling/TrainingSet.csv",index=False)
    df_testSet.to_csv(odirName+"/over_sampling/TestSet.csv",index=False)
    
if __name__ == "__main__":
    main()