import os
import glob
import argparse
import string
import warnings
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def get_args():
    parser = argparse.ArgumentParser(description="This script makes combines and cleans the positive and negative samples extracted.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
     
    parser.add_argument("--input","-i",type=str, default="/data/raw/Firefox", required = False, help="path to directory containing raw requirement data files")
    parser.add_argument("--output","-o",type=str, default="/data/preprocessed/firefox_data.csv", required=False,help="path to save the requirement combinations file")
    args = parser.parse_args()
    return args

def remove_punctuation(s):
    s = ''.join([i if i not in frozenset(string.punctuation) else ' ' for i in s])
    return s.lower()

def main():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    args=get_args()
    
    idir = args.input
    ofileName = args.output
    
    cols = ['req1Id','req1','req2Id','req2','BinaryClass','MultiClass']

    rqmt_combinations = pd.DataFrame(columns=cols)
    for filename in (glob.glob(os.getcwd()+idir+"/*.csv")):
        temp = pd.read_csv(filename)
        temp = temp [cols]

        rqmt_combinations = pd.concat([rqmt_combinations,temp])
    
    ########Map Values in label column
    #no-requires (Independent) - 0
    #requires - 1
    #similar - 2
    #blocks - 3

    rqmt_combinations['Label'] = rqmt_combinations['MultiClass'].map({'norequires':0,'requires':1,'similar':2,'blocks':3}) 
    rqmt_combinations = shuffle(rqmt_combinations)
    rqmt_combinations['req_1'] = rqmt_combinations['req1'].apply(remove_punctuation)
    rqmt_combinations['req_2'] = rqmt_combinations['req2'].apply(remove_punctuation)
    rqmt_combinations['comboId'] = range(0,len(rqmt_combinations))
    rqmt_combinations = rqmt_combinations[['comboId','req1Id','req1','req_1','req2Id','req2','req_2','Label']]
    rqmt_combinations.to_csv(os.getcwd()+ofileName, sep=',', index = True, encoding='utf-8')  #updated index = False, so that index values are not saved in csv, as everytime we read the file, index values will be generated by default.
    
    #save data to csv file
    print ("\nOuput file saved at : ",os.getcwd()+ofileName)
    
if __name__ == '__main__':
    main()