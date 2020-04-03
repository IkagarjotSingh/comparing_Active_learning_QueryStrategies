import pandas as pd
import numpy as np
import logs
from scipy.stats import entropy

def leastConfidenceSampling(df_uncertain):
    
    df_uncertain['lconf']=1-df_uncertain['maxProb']
    df_uncertain = df_uncertain.sort_values(by=['lconf'],ascending=False)
    #logs.writeLog("\n\nLeast Confidence Calculations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #logs.writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex

def minMarginSampling(df_uncertain):
    
    df_uncertain['sorted'] = df_uncertain['predictedProb'].sort_values().apply(lambda x:sorted(x,reverse=True))
    df_uncertain['first'] = [x[0] for x in df_uncertain['sorted']]
    df_uncertain['second'] = [x[1] for x in df_uncertain['sorted']] 
    df_uncertain['Margin'] = df_uncertain['first']-df_uncertain['second']
    
    df_uncertain = df_uncertain.sort_values(by=['Margin'],ascending=True)
    logs.writeLog("\n\nMin Margin Calcuations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #logs.writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex

def entropySampling(df_uncertain):
    df_uncertain['entropy'] = [entropy(x) for x in df_uncertain['predictedProb']]
    df_uncertain = df_uncertain.sort_values(by=['entropy'],ascending=False)
    #logs.writeLog("\n\nEntropy Calculations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #logs.writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex