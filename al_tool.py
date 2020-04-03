import os
import numpy as np
import pandas as pd
import warnings
from sklearn.utils import shuffle
import logs
import clf_model
import annotate
import splitData

warnings.filterwarnings("ignore")
    
def learnTargetLabel(args):
    '''
    Active Learning iterative process
    1. Prepare Data
    2. Create Classifier
    3. Evaluate Classifier
    4. Select Uncertain Samples and get them annotated by Oracle
    5. Update Data Set (Merge newly annotated samples to original dataset) 
    6. Repeat steps 1-5 until stopping condition is reached.

    Parameters : 
    args (dataframe) : Run-time arguments in a dataframe.

    Returns :
    df_rqmts (dataframe) : Updated / Final requirements dataset, included the prediction values at the last iteration of Active learning process. 
    df_resultTracker (dataframe) : Results for tracking purpose

    '''
    #Read run time arguments
    idir = os.getcwd()+args.loc[0,'input']    
    splitratio = float(args.loc[0,'testsize']) 
    maxIterations = int(args.loc[0,'maxIterations'])
    
    logs.writeLog("Fetching data from the input directory.")
    #Read To be Annotated, Training, Test and Validation Sets generated after executing splitData.py
    try:
        df_tobeAnnotated = pd.read_csv(idir+"/ToBeAnnotated.csv")
        df_training = pd.read_csv(idir+"/TrainingSet.csv")
        df_test = pd.read_csv(idir+"/TestSet.csv")
        df_manuallyAnnotated = pd.concat([df_training,df_test])
        df_validation = pd.read_csv(idir+"/ValidationSet.csv")
    except FileNotFoundError as err:
        logs.writeLog ("File Not Found! Please provide correct path of the directory containing Training, Test, Validation, ToBeAnnotated and Manually Annotated DataSet.")
        print (err)
        exit

    #Combines all requirement combinations in a single DataFrame
    df_rqmts = pd.concat([df_manuallyAnnotated,df_tobeAnnotated])
    
    
    #Create a dataframe to track the results
    df_resultTracker = pd.DataFrame(columns=['Iteration','ManuallyAnnotated','ToBeAnnotated','TrainingSize','TestSize','ValidationSize','ClassifierTestScore','ClassifierValidationScore','IndependentCount','RequiresCount','SimilarCount','BlocksCount','t_5FoldCVScore','t_10FoldCVScore','t_f1Score','t_precisionScore','t_recallScore','v_f1Score','v_precisionScore','v_recallScore','v_5FoldCVScore','v_10FoldCVScore'])
    
    iteration = 0
    
    while True:
        iteration+=1
        logs.writeLog("\n"+100*"-")
        logs.writeLog("\n\nIteration : "+str(iteration)+"\n")

        #For first iteration of active learning use the data available in df_train, df_test. 
        #For subsequent iterations, recreate the training and test sets as new data points will be annotated by manual annotator at end of each iteration.
        if iteration > 1 :
            df_manuallyAnnotated = df_rqmts[df_rqmts['AnnotationStatus']=='M'] #Training Data
            df_manuallyAnnotated['Label'] = df_manuallyAnnotated['Label'].astype('int')
    
            logs.writeLog("\nSplitting the Training/Test Set into training and test set - "+str(1-splitratio)+"/"+str(splitratio)+" split.")
            df_training,df_test = splitData.balanceDataSet(df_manuallyAnnotated,"Label",1-splitratio)
            
            df_tobeAnnotated = df_rqmts[df_rqmts['AnnotationStatus']!='M']
            
        logs.writeLog("\nCreating Classifier...")
        countVectorizer, tfidfTransformer, classifier, classifierTestScore,t_f1Score,t_precisionScore,t_recallScore = clf_model.createClassifier(args.loc[0,'classifier'],df_training,df_test)  
        
        logs.writeLog ("\n\nEvaluating 5 fold and 10 fold Cross Validation Scores (Test Set)...")
        t_cf5_score,t_cf10_score = clf_model.Crossfoldvalidation(countVectorizer,tfidfTransformer,classifier,pd.concat([df_training,df_test]))
        
        logs.writeLog("\n\n5 fold Cross Validation Score : "+str(t_cf5_score))
        logs.writeLog("\n\n10 fold Cross Validation Score : "+str(t_cf10_score))
        
        logs.writeLog ("\n\nValidating Classifier...")
        classifierValidationScore,v_f1Score,v_precisionScore,v_recallScore = clf_model.validateClassifier(countVectorizer,tfidfTransformer,classifier,df_validation)
        logs.writeLog("\n\nClassifier Validation Set Score : "+str(classifierValidationScore))
        
        logs.writeLog ("\n\nEvaluating 5 fold and 10 fold Cross Validation Scores (Validation Set)...")
        v_cf5_score,v_cf10_score = clf_model.Crossfoldvalidation(countVectorizer,tfidfTransformer,classifier,df_validation)
        
        logs.writeLog("\n\n5 fold Cross Validation Score : "+str(v_cf5_score))
        logs.writeLog("\n\n10 fold Cross Validation Score : "+str(v_cf10_score))
        
        #Update Analysis DataFrame (For tracking purpose)
        df_training['Label'] = df_training['Label'].astype('int')
        df_test['Label'] = df_test['Label'].astype('int')
        independentCount = len(df_training[df_training['Label']==0]) + len(df_test[df_test['Label']==0])
        requiresCount = len(df_training[df_training['Label']==1]) + len(df_test[df_test['Label']==1])
        similarCount = len(df_training[df_training['Label']==2]) + len(df_test[df_test['Label']==2])
        blocksCount = len(df_training[df_training['Label']==3]) + len(df_test[df_test['Label']==3])
        
        df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':len(df_manuallyAnnotated),'ToBeAnnotated':len(df_tobeAnnotated),'TrainingSize':len(df_training),'TestSize':len(df_test),'ValidationSize':len(df_validation),'ClassifierTestScore':classifierTestScore,'ClassifierValidationScore':classifierValidationScore,'IndependentCount':independentCount,'RequiresCount':requiresCount,'SimilarCount':similarCount,'BlocksCount':blocksCount,'t_5FoldCVScore':t_cf5_score,'t_10FoldCVScore':t_cf10_score,'t_f1Score':t_f1Score,'t_precisionScore':t_precisionScore,'t_recallScore':t_recallScore,'v_5FoldCVScore':v_cf5_score,'v_10FoldCVScore':v_cf10_score,'v_f1Score':v_f1Score,'v_precisionScore':v_precisionScore,'v_recallScore':v_recallScore},ignore_index=True)

        logs.writeLog("\n\nAnalysis DataFrame : \n"+str(df_resultTracker))
        
        logs.writeLog ("\n\nPredicting Labels....")
        df_predictionResults = clf_model.predictLabels(countVectorizer,tfidfTransformer,classifier,df_tobeAnnotated)  
        
        logs.writeLog("\n\nFinding Uncertain Samples and Annotating them.....")
        df_finalPredictions = annotate.analyzePredictions(args,df_predictionResults)
        
        logs.writeLog("\n\nMerging Newly Labelled Data Samples....")
        df_rqmts = pd.concat([df_training,df_test,df_finalPredictions],axis=0,ignore_index=True)
        #Remove unwanted columns
        df_rqmts = df_rqmts[['comboId','req1Id','req1','req_1','req2Id','req2','req_2','Label','AnnotationStatus']]
        
        if iteration >=maxIterations:
            logs.writeLog("\n\nStopping Condition Reached... Exiting the program.")
            break
        
        
    #Merge Validation Set back to the prediction set to ensure all the 19699 combinations are returned.
    df_rqmts = pd.concat([df_rqmts,df_validation],axis=0,ignore_index=True)
    
    return df_rqmts,df_resultTracker
        
def main():
    #Ignore Future warnings if any occur. 
    warnings.simplefilter(action='ignore', category=FutureWarning)  
    
    pd.set_option('display.max_columns', 500)   #To make sure all the columns are visible in the logs.
    pd.set_option('display.width', 1000)

    #initialize directory which contains all the data and which will contain logs and outputs
    currentFileDir = os.getcwd()
    
    #Reads run time arguments
    args = logs.getArguments(currentFileDir+"/ALParams.txt") 
    comments = args.loc[0,'comments']

    #Creates Logs folder structure
    logFilePath,OFilePath = logs.createLogs(currentFileDir+"/Logs",args)   
    
    df_rqmtComb,df_Analysis = learnTargetLabel(args)
    
    #Adds the Analysis DataFrame to Output File
    logs.addOutputToExcel(df_Analysis,"\nAnalysis of  Label Classification  : \n")
    
    logs.updateResults(df_rqmtComb,args)   #Update Results in excel....
    
    logs.writeLog("\nOutput Analysis is available at : "+str(OFilePath))
    logs.writeLog("\nLogs are available at : "+str(logFilePath))
    
if __name__ == '__main__':
    main()