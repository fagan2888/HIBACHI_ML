#!/usr/bin/env python
#==============================================================================
#
#          FILE:  evals.py
# 
#         USAGE:  import evals (from hib.py)
# 
#   DESCRIPTION:  evaluation routines
# 
#       UPDATES:  170339: removed shuffle from getfolds()
#                 170410: added reclass()
#                 170417: renamed reclass() to reclass_result()
#                         reworked reclass_result()
#                 170510: reclass_result() convert result to numpy array before
#                         attaching to pandas DataFrame
#        AUTHOR:  Pete Schmitt (hershey), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.3
#       CREATED:  Sun Mar 19 11:34:09 EDT 2017
#      REVISION:  Wed May 10 15:28:06 EDT 2017
#==============================================================================
import numpy as np
import pandas as pd
import sys
###############################################################################
def subsets(x,percent):
    """ take a subset of "percent" of x """
    p = percent / 100
    xa = np.array(x)
    subsample_indices = np.random.choice(xa.shape[1], int(xa.shape[1] * p), 
                                         replace=False)
    return (xa[:, subsample_indices]).tolist()
###############################################################################
def getfolds(x, num):
    """ return num folds of size 1/num'th of x """
    folds = []
    fsize = end = int(len(x[0]) / num)
    xa = np.array(x)
    start = 0
    for i in range(num):
        folds.append(xa[:,start:end])
        start += fsize
        end += fsize  
    return folds
###############################################################################
def addnoise(x,pcnt):
    """ add some percentage of noise to data """
    xa = np.array(x)
    val = pcnt/100
    rep = {}
    rep[0] = [1,2]
    rep[1] = [0,2]
    rep[2] = [0,1]

    for i in range(len(xa)):
        indices = np.random.choice(xa.shape[1], int(xa.shape[1] * val), 
                                   replace=False)
        for j in list(indices):
            xa[i][j] = np.random.choice(rep[xa[i][j]])

    return xa.tolist()
###############################################################################
def reclass_result(x, result, pct):
    """ reclassify data """
    d = np.array(x).transpose()
    columns = [0]*np.shape(d)[1] 
    for i in range(0,np.shape(d)[1]): # create columns names for variable number of columns.
        columns[i] = 'X' + str(i)
    df = pd.DataFrame(d, columns=columns)
#    df = pd.DataFrame(d, columns=['X0','X1','X2'])
    dflen = len(df)
    np_result = np.array(result)

    df['Class'] = np_result

    df.sort_values('Class', ascending=True, inplace=True)
    
    cntl_cnt = dflen - int(dflen * (pct/100.0))
    c = np.zeros(dflen, dtype=np.int)
    c[cntl_cnt:] = 1

    df.Class = c
    df.sort_index(inplace=True)  # put data back in index order
    return df['Class'].tolist()
