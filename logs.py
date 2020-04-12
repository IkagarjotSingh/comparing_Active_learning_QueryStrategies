import datetime
import os
import pandas as pd

def createLogs(fPath,args):
    '''
    Creates the structure for saving the log file, output file, results file and annotations file
    1. Log file - saves all the details are printed on the command line
    2. Output File - saves all outputs (Analysis Dataframes in this case)
    3. Results File - saves all predicted labels of unlabelled dataset
    4. Annotations File - saves all manually annotated data points (by oracle)
    '''
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    
    if not os.path.exists(fPath+"/"+current_date):
        os.makedirs(fPath+"/"+current_date)
    global logFilePath,outputFilePath,resultsPath,annotationsPath
    logFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".txt"
    outputFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".csv"
    resultsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-RESULTS-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".csv"
    annotationsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-ANNOTATIONS-"+args.loc[0,'resampling']+"_"+args.loc[0,'comments']+".csv"
    for fPath in [logFilePath,outputFilePath]:
        file = open(fPath,'a')
        file.write("\n"+100*"-"+"\nArguments :- \n")
        for col in args.columns:
            file.write(str(col)+" : "+str(args.loc[0,str(col)])+"\n")
        file.write("\n"+100*"-"+"\n")
        file.close()

    
    return logFilePath,outputFilePath


def getArguments(fName):
    '''
    Reads the arguments available in the file and converts them into a data frame.
    '''
    file = open(fName,'r')
    df_args = pd.DataFrame()
    print ("\n"+100*"-"+"\nArguments :- \n")
    for line in file:
        print (line.strip())
        kv_pair = line.split(":")
        df_args.loc[0,str(kv_pair[0]).strip()] = str(kv_pair[1]).strip()
    print (100*"-")
    validateArguments(df_args)
    return df_args

def validateArguments(df_args):
    '''
    Validates the arguments.
    '''
    try:
        if not os.path.exists(os.getcwd()+df_args.loc[0,'input']):
            raise("")
        elif ((df_args.loc[0,'classifier'] not in ['RF','NB','SVM','ensemble']) or (df_args.loc[0,'resampling'] not in ['under_sampling','over_sampling'])or (df_args.loc[0,'samplingType'] not in ['leastConfidence','minMargin','entropy']) ):
            raise ("")
        elif (float(df_args.loc[0,'testsize']) not in [x/10 for x in range(0,11)]):
            raise ("")
        elif ((int(df_args.loc[0,'manualAnnotationsCount']))or (int(df_args.loc[0,'maxIterations']))):
            pass
    except :
        print ("\nERROR! Input Arguments are invalid....\nPlease verify your values with following reference.\n")
        showExpectedArguments()
        exit()
    return None

def showExpectedArguments():
    '''
    prints the expected arguments, stored at ALParams_Desc.txt 
    '''
    file = open(os.getcwd()+"/ALParams_Desc.txt")
    for line in file:
        print (line)

def writeLog(content):
    '''
    Dumps the content into Log file
    '''
    file = open(logFilePath,"a", encoding='utf-8')
    file.write(content)
    file.close()
    print (content)
    return None


def createAnnotationsFile(df_rqmts):
    '''
    Dumps the manuall Annotations data into a csv file.
    '''
    if not os.path.exists(annotationsPath):
        df_rqmts.to_csv(annotationsPath,mode="a",index=False,header=True)
    else:
        df_rqmts.to_csv(annotationsPath,mode="a",index=False,header=False)
    return resultsPath


def addOutputToExcel(df,comment):
    '''
    Appends the dataframe df and corresponding comment to the output file.
    '''
    file = open(outputFilePath,"a", encoding='utf-8')
    file.write(comment)
    file.close()
    print (comment)
    print (str(df))
    df.to_csv(outputFilePath,mode='a',index=False)
    return None

def updateResults(df_results,args):
    '''
    Merges the Results data frame with arguments dataframe and stores the results in a csv file. 
    '''
    df_results.reset_index(inplace=True,drop=True)
    args.reset_index(inplace=True,drop=True)
    combined_df = pd.concat([df_results,args],axis=1)
    combined_df.to_csv(resultsPath,mode="a",index=False)
    
    return resultsPath
