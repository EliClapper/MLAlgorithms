# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:26:01 2022

@author: e_lib
"""
import numpy as np
import pandas as pd

np.random.seed(6164900)
y = pd.Series(np.random.normal(0,1,100))
x = pd.Series(y + np.random.normal(0,3,100))
df = pd.concat([x,y], axis = 1, keys=['y', 'x'])

# Calculate Eucledian Distance between two Series
def EucDist(C,N):
    return((((C-N)**2).sum())**0.5)
# get Eucledian distance of first case with all other cases
    
def PredictSingleCase(caseindex, df, K):
    EucDists = []
    df_wo_case = df.drop(index = caseindex)                       # create df without the specific case we would like to predict
    for i in [x for x in list(range(len(df))) if x != caseindex]: # for i in list of 1 to n where caseindex is not included
        EucDists.append(EucDist(df.loc[caseindex], df_wo_case.loc[i])) # get all euclidean distances for that case
    KNNs = pd.Series(EucDist) #make Series of the Euclidean Distances
    KNNs.index = [x for x in list(range(len(df))) if x != caseindex] #make index numbers true to df_wo_case so that we can subset df_wo_case for the correct indices
    KNNs = KNNs.sort_values()[:K] #sort ascending where only K neighbours with the lowest ED's are retained
    return(df_wo_case.loc[KNNs.index]['y'].mean()) # return the mean of the K neighbours on the outcome variable which is the predicted value for the case

### Now only the function remains that does the actual KNN.

    