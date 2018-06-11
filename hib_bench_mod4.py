# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:06:59 2017

@author: Max
"""

#!/usr/bin/env python3
#===============================================================================
#
#          FILE:  hib.py
# 
#         USAGE:  ./hib.py [options]
# 
#   DESCRIPTION:  Data simulation software that creates data sets with 
#                 particular characteristics
#
#       OPTIONS:  ./hib.py -h for all options
#
#  REQUIREMENTS:  python >= 3.5, deap, scikit-mdr, pygraphviz
#          BUGS:  Damn ticks!!
#       UPDATES:  170224: try/except in evalData()
#                 170228: files.sort() to order files
#                 170313: modified to use IO.get_arguments()
#                 170319: modified to use evals for evaluations
#                 170320: modified to add 1 to data elements before processing
#                 170323: added options for plotting
#                 170410: added call to evals.reclass_result() in evalData()
#                 170417: reworked post processing of new random data tests
#                 170422: added ability for output directory selection
#                         directory is created if it doesn't exist
#                 170510: using more protected operators from operators.py
#       AUTHORS:  Pete Schmitt (discovery), pschmitt@upenn.edu
#                 Randy Olson, olsonran@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.10
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Wed May 10 12:41:21 EDT 2017
#===============================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
import re
#===============================================================================
from deap import algorithms, base, creator, tools, gp
from mdr.utils import three_way_information_gain as three_way_ig
from mdr.utils import two_way_information_gain as two_way_ig
import IO
import evals
import itertools
import glob
import numpy as np
import operator as op
import operators as ops
import os
import pandas as pd
import random
import sys
import time
###############################################################################
if (sys.version_info[0] < 3):
    print("hibachi requires Python version 3.5 or later")
    sys.exit(1)

labels = []
all_igsums = []
#results = []
start = time.time()

options = IO.get_arguments()
infile = options['file']
evaluate = options['evaluation']
population = options['population']
generations = options['generations']
rdf_count = options['random_data_files']
ig = options['information_gain']
rows = options['rows']
cols = options['columns']
Stats = options['statistics']
Trees = options['trees']
Fitness = options['fitness']
prcnt = options['percent']
outdir = options['outdir']
up_method = options['up']
down_method = options['down']
if Fitness or Trees or Stats:
    import plots
#
# set up random seed
#
######################################################################
if options['up'] == options['down']:
    print("Please choose different ML functions for up & down")
    sys.exit(1)
######################################################################

if(options['seed'] == -999):
    rseed = random.randint(1,1000)
else:
    rseed = options['seed']
random.seed(rseed)
np.random.seed(rseed)
#
# Read/create the data and put it in a list of lists.
# data is normal view of columns as features
# x is transposed view of data
#
if infile == 'random':
    data, x = IO.get_random_data(rows,cols,rseed)
else:
    data, x = IO.read_file(infile)
    rows = len(data)
    cols = len(x)

inst_length = len(x)
###############################################################################
# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, inst_length), 
                            float, "X")
# basic operators 
pset.addPrimitive(ops.addition, [float,float], float)
pset.addPrimitive(ops.subtract, [float,float], float)
pset.addPrimitive(ops.multiply, [float,float], float)
pset.addPrimitive(ops.safediv, [float,float], float)
pset.addPrimitive(ops.modulus, [float,float], float)
pset.addPrimitive(ops.plus_mod_two, [float,float], float)
# logic operators 
pset.addPrimitive(ops.gt, [float, float], float)
pset.addPrimitive(ops.lt, [float, float], float)
pset.addPrimitive(ops.AND, [float, float], float)
pset.addPrimitive(ops.OR, [float, float], float)
pset.addPrimitive(ops.xor, [float,float], float)
# bitwise operators 
pset.addPrimitive(ops.bitand, [float,float], float)
pset.addPrimitive(ops.bitor, [float,float], float)
pset.addPrimitive(ops.bitxor, [float,float], float)
# unary operators 
pset.addPrimitive(op.abs, [float], float)
pset.addPrimitive(ops.NOT, [float], float)
pset.addPrimitive(ops.factorial, [float], float)
pset.addPrimitive(ops.left, [float,float], float)
pset.addPrimitive(ops.right, [float,float], float)
# large operators 
pset.addPrimitive(ops.power, [float,float], float)
pset.addPrimitive(ops.logAofB, [float,float], float)
pset.addPrimitive(ops.permute, [float,float], float)
pset.addPrimitive(ops.choose, [float,float], float)
# misc operators 
pset.addPrimitive(min, [float,float], float)
pset.addPrimitive(max, [float,float], float)
# terminals 
randval = "rand" + str(random.random())[2:]  # so it can rerun from ipython
pset.addEphemeralConstant(randval, lambda: random.random() * 100, float)
pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float)
# creator 
creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0,1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
# toolbox 
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual",
                 tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
##############################################################################
def evalData(individual, xdata, xtranspose):
    """ evaluate the individual """
    result = []
    igsums = np.array([])
    x = xdata
    data = xtranspose

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    tree = str(individual)
    #num features:
##############################################################################    
    list_matches = [(m.start(0), m.end(0)) for m in re.finditer(r'X[0123456789]', tree)]
    list_of_matches = []
    for i in list_matches:
        list_of_matches.append(tree[i[0]:i[1]])
    num_features = len(set(list_of_matches))
##############################################################################
    # Create class possibility.  
    # If class has a unique length of 1, toss it.
    try:
        result = [(func(*inst[:inst_length])) for inst in data]
        dist = result
    except:
        labels.append((0, 0, 0))
        return -sys.maxsize, -sys.maxsize, -sys.maxsize

    if (len(np.unique(result)) == 1):
        labels.append((0, 0, 0))
        return -sys.maxsize, -sys.maxsize, -sys.maxsize
    
     
    if evaluate == 'normal':
        rangeval = 1

    elif evaluate == 'folds':
        rangeval = numfolds = 10  # must be equal
        folds = evals.getfolds(x, numfolds)

    elif evaluate == 'subsets':
        rangeval = 10
        percent = 25

    elif evaluate == 'noise':
        rangeval = 10
        percent = 10
    
    if num_features <= 3:
        return -sys.maxsize, -sys.maxsize, -sys.maxsize
        
    result = np.array(evals.reclass_result(x, result, prcnt))
    y = result
    
#    print(result.shape)

#    result = np.array(result).T
#   results.append(result)

    for m in range(rangeval):
        igsum = 0 
        if evaluate == 'folds': 
            xsub = list(folds[m])

        elif evaluate == 'subsets': 
            xsub = evals.subsets(x,percent)

        elif evaluate == 'noise': 
            xsub = evals.addnoise(x,percent)

        else:  # normal
            xsub = np.array(x).T
    x = data
#    print(np.shape(y),np.shape(x))
        # Calculate information gain between data columns and result
        # and return mean of these calculations
#        if(ig == 2):
#            for i in range(inst_length):
#                for j in range(i+1,inst_length):
#                    igsum += two_way_ig(xsub[i], xsub[j], result)
#        elif(ig == 3):
#            for i in range(inst_length):
#                for j in range(i+1,inst_length):
#                    for k in range(j+1,inst_length):
#                        igsum += three_way_ig(xsub[i], xsub[j], xsub[k], result)
#                    
#        igsums = np.append(igsums,igsum)

    
        
#    igsum_avg = np.mean(igsums)
#    labels.append((igsum_avg, result)) # save all results
#    all_igsums.append(igsums)
######################################################
######################################################
######################################################
#    test_score_dt = np.array(range(5))
#    test_score_lr_l1 = np.array(range(5))
#    test_score_svm = np.array(range(5))
#    for i in range(0,5):
#        X_train, X_test, y_train, y_test = ms.train_test_split(xsub, result.T, test_size=0.2)
#        #####################################################
##        svm_model = SVC()
##        svm_model.fit(X_train, y_train)
###        y_pred_test = dt_model.predict(X_test)
##        test_score_svm[i] = svm_model.score(X_test, y_test)
##        del svm_model
#        #####################################################
#        dt_model = DecisionTreeClassifier(max_depth=5)
#        dt_model.fit(X_train, y_train)
#    #    y_pred_test = dt_model.predict(X_test)
#        test_score_dt[i] = dt_model.score(X_test, y_test)
#        del dt_model
#        #####################################################
#        lr_model = LogisticRegression(solver='liblinear', penalty='l1')
#        lr_model.fit(X_train, y_train)
#    #    y_pred_test = lr_model.predict(X_test)
#        test_score_lr_l1[i] = lr_model.score(X_test, y_test)
#        del lr_model
#        #####################################################
##########################################   
##########################################   
##########################################    
#    up = np.mean(test_score_dt)
#    down = np.mean(test_score_lr_l1)
##    up = np.mean(test_score_svm)
##    down = np.mean(test_score_dt)
##    up = np.mean(test_score_lr_l1)
##########################################    
##########################################   
##########################################   
######################################################
######################################################
######################################################
    x = np.array(x)
    y = np.array(y)    
    auc_up = [0]*10
    auc_down = [0]*10
    m1 = [0]*10
    m2 = [0]*10
    m3 = [0]*10
    m4 = [0]*10
    methods = [0]*6
    
    if options['up'] == 'svm':
        methods[0] = 1
    elif options['up'] == 'dt':
        methods[1] = 1
    elif options['up'] == 'lr2':
        methods[2] = 1
    elif options['up'] == 'rf':
        methods[3] = 1
    elif options['up'] == 'gb':
        methods[4] = 1
    elif options['up'] == 'knn':
        methods[5] = 1

    if options['down'] == 'svm':
        methods[0] = 1
    elif options['up'] == 'dt':
        methods[1] = 1
    elif options['up'] == 'lr2':
        methods[2] = 1
    elif options['up'] == 'rf':
        methods[3] = 1
    elif options['up'] == 'gb':
        methods[4] = 1
    elif options['up'] == 'knn':
        methods[5] = 1
    
    roc_auc_for_std = []
    
    for u in range(10):
        X_train, X_test, y_train, y_test = ms.train_test_split(x,y, test_size=0.2)
    
    
    #    y_hat2 = lr2_model.predict_proba(X_test)[:,1]
    #    fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
    #    roc_auc_down = metrics.auc(fpr2, tpr2)
        #print(type(X_train))
        
        if options['up'] == 'svm':
            svm_model = SVC(probability=True)
            svm_model.fit(X_train,y_train)
            y_hat1 = svm_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat1, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_up = metrics.auc(fpr2, tpr2)
    #        cv5_score_up = np.mean(cross_val_score(svm_model, x, y, cv=5))
            del svm_model
        elif options['up'] == 'dt':
            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train,y_train)
            y_hat1 = dt_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat1, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_up = metrics.auc(fpr2, tpr2)
    #        cv5_score_up = np.mean(cross_val_score(dt_model, x, y, cv=5))
            del dt_model
        # elif options['up'] == 'lr1':
            # lr1_model = LogisticRegression(solver='liblinear', penalty='l1')
            # lr1_model.fit(X_train,y_train)
            # y_hat1 = lr1_model.predict_proba(X_test)[:,1]
            # fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat1, pos_label=1, sample_weight=None, drop_intermediate=True)
            # roc_auc_up = metrics.auc(fpr2, tpr2)
    # #        cv5_score_up = np.mean(cross_val_score(lr1_model, x, y, cv=5))
            # del lr1_model
        elif options['up'] == 'lr2':
            lr2_model = LogisticRegression(solver='liblinear', penalty='l2')
            lr2_model.fit(X_train,y_train)
            y_hat1 = lr2_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat1, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_up = metrics.auc(fpr2, tpr2)
    #        cv5_score_up = np.mean(cross_val_score(lr2_model, x, y, cv=5))
            del lr2_model
        elif options['up'] == 'rf':
            rf_model = RandomForestClassifier(n_estimators=100)
            rf_model.fit(X_train,y_train)
            y_hat1 = rf_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat1, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_up = metrics.auc(fpr2, tpr2)
    #        cv5_score_up = np.mean(cross_val_score(rf_model, x, y, cv=5))
            del rf_model
        elif options['up'] == 'gb':
            gb_model = GradientBoostingClassifier(n_estimators=10)
            gb_model.fit(X_train,y_train)
            y_hat1 = gb_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat1, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_up = metrics.auc(fpr2, tpr2)
    #        cv5_score_up = np.mean(cross_val_score(gb_model, x, y, cv=5))
            del gb_model
        elif options['up'] == 'knn':
            knn_model = KNeighborsClassifier()
            knn_model.fit(X_train,y_train)
            y_hat1 = knn_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat1, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_up = metrics.auc(fpr2, tpr2)
    #        cv5_score_up = np.mean(cross_val_score(knn_model, x, y, cv=5))
            del knn_model
        else:
            print("Wrong ML function name. Supported functions: svm, dt, rf, lr2, gb, xgb, knn")
            sys.exit(1)
#####################################################################################################################################################################################                   
        if options['down'] == 'svm':
            svm_model = SVC(probability=True)
            svm_model.fit(X_train,y_train)
            y_hat2 = svm_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_down = metrics.auc(fpr2, tpr2)
    #        cv5_score_down = np.mean(cross_val_score(svm_model, x, y, cv=5))
            del svm_model
        elif options['down'] == 'dt':
            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train,y_train)
            y_hat2 = dt_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_down = metrics.auc(fpr2, tpr2)
    #        cv5_score_down = np.mean(cross_val_score(dt_model, x, y, cv=5))
            del dt_model
        # elif options['down'] == 'lr1':
            # lr1_model = LogisticRegression(solver='liblinear', penalty='l1')
            # lr1_model.fit(X_train,y_train)
            # y_hat2 = lr1_model.predict_proba(X_test)[:,1]
            # fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            # roc_auc_down = metrics.auc(fpr2, tpr2)
    # #        cv5_score_down = np.mean(cross_val_score(lr1_model, x, y, cv=5))
            # del lr1_model
        elif options['down'] == 'lr2':
            lr2_model = LogisticRegression(solver='liblinear', penalty='l2')
            lr2_model.fit(X_train,y_train)
            y_hat2 = lr2_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_down = metrics.auc(fpr2, tpr2)
    #        cv5_score_down = np.mean(cross_val_score(lr2_model, x, y, cv=5))
            del lr2_model
        elif options['down'] == 'rf':
            rf_model = RandomForestClassifier(n_estimators=100)
            rf_model.fit(X_train,y_train)
            y_hat2 = rf_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_down = metrics.auc(fpr2, tpr2)
    #        cv5_score_down = np.mean(cross_val_score(rf_model, x, y, cv=5))
            del rf_model
        elif options['down'] == 'gb':
            gb_model = GradientBoostingClassifier() # n_estimators=10
            gb_model.fit(X_train,y_train)
            y_hat2 = gb_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_down = metrics.auc(fpr2, tpr2)
    #        cv5_score_down = np.mean(cross_val_score(gb_model, x, y, cv=5))
            del gb_model
        elif options['down'] == 'knn':
            knn_model = KNeighborsClassifier()
            knn_model.fit(X_train,y_train)
            y_hat2 = knn_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_down = metrics.auc(fpr2, tpr2)
    #        cv5_score_down = np.mean(cross_val_score(knn_model, x, y, cv=5))
            del knn_model
        else:
            print("Wrong ML function name. Supported functions: svm, dt, rf, lr2, gb, knn")
            sys.exit(1)
#####################################################################################################################################################################################       
        if methods[0] == 0:
            svm_model = SVC(probability=True)
            svm_model.fit(X_train,y_train)
            y_hat2 = svm_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_for_std.append(metrics.auc(fpr2, tpr2))
    #        cv5_score_down = np.mean(cross_val_score(svm_model, x, y, cv=5))
            del svm_model
        if methods[1] == 0:
            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train,y_train)
            y_hat2 = dt_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_for_std.append(metrics.auc(fpr2, tpr2))
    #        cv5_score_down = np.mean(cross_val_score(dt_model, x, y, cv=5))
            del dt_model
        if methods[2] == 0:
            lr2_model = LogisticRegression(solver='liblinear', penalty='l2')
            lr2_model.fit(X_train,y_train)
            y_hat2 = lr2_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_for_std.append(metrics.auc(fpr2, tpr2))
    #        cv5_score_down = np.mean(cross_val_score(lr2_model, x, y, cv=5))
            del lr2_model
        if methods[3] == 0:
            rf_model = RandomForestClassifier(n_estimators=100)
            rf_model.fit(X_train,y_train)
            y_hat2 = rf_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_for_std.append(metrics.auc(fpr2, tpr2))
    #        cv5_score_down = np.mean(cross_val_score(rf_model, x, y, cv=5))
            del rf_model
        if methods[4] == 0:
            gb_model = GradientBoostingClassifier() # n_estimators=10
            gb_model.fit(X_train,y_train)
            y_hat2 = gb_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_for_std.append(metrics.auc(fpr2, tpr2))
    #        cv5_score_down = np.mean(cross_val_score(gb_model, x, y, cv=5))
            del gb_model
        if methods[5] == 0:
            knn_model = KNeighborsClassifier()
            knn_model.fit(X_train,y_train)
            y_hat2 = knn_model.predict_proba(X_test)[:,1]
            fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_hat2, pos_label=1, sample_weight=None, drop_intermediate=True)
            roc_auc_for_std.append(metrics.auc(fpr2, tpr2))
    #        cv5_score_down = np.mean(cross_val_score(knn_model, x, y, cv=5))
            del knn_model
            
            
        m1[u] = roc_auc_for_std[0]
        m2[u] = roc_auc_for_std[1]
        m3[u] = roc_auc_for_std[2]
        m4[u] = roc_auc_for_std[3]
        auc_up[u] = roc_auc_up
        auc_down[u] = roc_auc_down
    m1_all = np.mean(m1)
    m2_all = np.mean(m2)
    m3_all = np.mean(m3)
    m4_all = np.mean(m4)
    # print(m1_all)
    # print(m2_all)
    # print(m3_all)
    # print(m4_all)
    
    roc_auc_std = np.std([m1_all,m2_all,m3_all,m4_all,roc_auc_up,roc_auc_down])
    roc_auc_up = np.mean(auc_up)
    roc_auc_down = np.mean(auc_down)
#    result = list(result.T)
    labels.append((roc_auc_up - roc_auc_down, result, dist)) # save all results
    
    if len(individual) <= 1:
        return -sys.maxsize, -sys.maxsize, -sys.maxsize
    else:
        return roc_auc_up, (roc_auc_up - roc_auc_down), roc_auc_std
#        if evaluate == 'normal':
#            return igsum, len(individual)
#        else:
#            return igsum_avg, len(individual)

##############################################################################
toolbox.register("evaluate", evalData, xdata = x, xtranspose=data)
toolbox.register("select", tools.selNSGA2)
toolbox.register("Tournament", tools.selDoubleTournament)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=op.attrgetter('height'), max_value=90))
toolbox.decorate("mutate", gp.staticLimit(key=op.attrgetter('height'), max_value=90))
toolbox.decorate("select", gp.staticLimit(key=op.attrgetter('height'), max_value=90))
##############################################################################
def pareto_eq(ind1, ind2):
    """Determines whether two individuals are equal on the Pareto front
       Parameters (ripped from tpot's base.py)
        ----------
        ind1: DEAP individual from the GP population
         First individual to compare
        ind2: DEAP individual from the GP population
         Second individual to compare
        Returns
        ----------
        individuals_equal: bool
         Boolean indicating whether the two individuals are equal on
         the Pareto front
    """
    return np.all(ind1.fitness.values == ind2.fitness.values)
##############################################################################
def hibachi(pop,gen,rseed):
    """ set up stats and population size,
        then start the process """
    MU, LAMBDA = pop, pop
    NGEN = gen 
    np.random.seed(rseed)
    random.seed(rseed)
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront(similar=pareto_eq)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    pop, log = algorithms.eaMuPlusLambda(pop,toolbox,mu=MU,lambda_=LAMBDA, 
                          cxpb=0.7, mutpb=0.3, ngen=NGEN, stats=stats, 
                          verbose=True, halloffame=hof)
    
    return pop, stats, hof, log

##############################################################################
# run the program
##############################################################################
print('input data:  ' + infile)
print('population:  ' + str(population))
print('generations: ' + str(generations))
print('evaluation:  ' + str(evaluate))
print('ign 2/3way:  ' + str(ig))
print('random seed: ' + str(rseed))
print('prcnt cases: ' + str(prcnt) + '%')
print('output dir:  ' + outdir)
print()
# Here we go...
pop, stats, hof, logbook = hibachi(population,generations,rseed)
best = []
fitness = []
for ind in hof:
    best.append(ind)
    fitness.append(ind.fitness.values)

for i in range(len(hof)):
    print("Best", i, "=", best[i])
    print("Fitness", i, '=', fitness[i])

record = stats.compile(pop)
print("statistics:")
print(record)

tottime = time.time() - start
if tottime > 3600:
    IO.printf("\nRuntime: %.2f hours\n", tottime/3600)
elif tottime > 60:
    IO.printf("\nRuntime: %.2f minutes\n", tottime/60)
else:
    IO.printf("\nRuntime: %.2f seconds\n", tottime)
df = pd.DataFrame(logbook)
del df['gen']
del df['nevals']
#
# sys.exit(0)
#
if(infile == 'random'):
    file1 = 'random0'
else:
    file1 = os.path.splitext(os.path.basename(infile))[0]
#
# make output directory if it doesn't exist
#
outdir = "results_mod4_test/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

Fitness = [i[0] for i in labels]
fitness = [np.max(Fitness[0:i+population]) for i in range(0, len(Fitness), population)]
labels.sort(key=op.itemgetter(0),reverse=True)     # sort by igsum (score)
tag = np.random.randint(100)
outfile = outdir + "AAA_bench_mod_4_up_down_" + up_method + "_" + down_method + "_score_" + str(round(labels[0][0],5)) + "_" + str(tag) + ".txt" 
print("writing data with Class to", outfile)

IO.create_file(x,labels[0][1],labels[0][2],fitness,outfile)       # use first individual
outfile2 = outdir + "III_bench_mod_4_up_down_" + up_method + "_" + down_method + "_score_" + str(round(labels[0][0],5)) + "_" + str(tag) + "_individual.txt" 
del tag
df_ind = best[0]
print(df_ind)
with open(outfile2, "w") as text_file:
    print(df_ind, file=text_file)
#
# test results against other data
#
if rdf_count == 0:
    files = glob.glob('data/in*')
    files.sort()
#
#  Test remaining data files with best individual
#
save_seed = rseed
if(infile == 'random' or rdf_count > 0):
    print('number of random data to generate:',rdf_count)
    for i in range(rdf_count):
        rseed += 1
        D, X = IO.get_random_data(rows,cols,rseed)
        nfile = 'random' + str(i+1)
        print(nfile)
        individual = best[0]
        func = toolbox.compile(expr=individual)
        result = [(func(*inst[:inst_length])) for inst in D]
        nresult = evals.reclass_result(X, result, prcnt)
        outfile = outdir + 'model_from-' + file1 
        outfile += '-using-' + nfile + '-' + str(rseed) + '-' 
        outfile += str(evaluate) + '-' + str(ig) + "way.txt" 
        print(outfile)
        IO.create_file(X,nresult,outfile)
else:
    print('number of files:',len(files))
    for i in range(len(files)):
        rseed += 1
        if files[i] == infile: continue
        nfile = os.path.splitext(os.path.basename(files[i]))[0]
        print(infile)
        print()
        D, X = IO.read_file(files[i]) #  new data file
        print('input file:', files[i])
        individual = best[0]
        func = toolbox.compile(expr=individual)
        result = [(func(*inst[:inst_length])) for inst in D]
        nresult = evals.reclass_result(X, result, prcnt)
        outfile = outdir + 'model_from-' + file1 + '-using-' + nfile + '-'
        outfile += str(rseed) + '-' + nfile + '-'
        outfile += str(evaluate) + '-' + str(ig) + "way.txt" 
        print(outfile)
        IO.create_file(X,nresult,outfile)
#
# plot data if selected
#
file = os.path.splitext(os.path.basename(infile))[0]
if Stats == True:
    statfile = outdir + "stats-" + file + "-" + evaluate 
    statfile += "-" + str(rseed) + ".pdf"
    print('saving stats to', statfile)
    plots.plot_stats(df,statfile)

if Trees == True:
    print('saving tree plot to ' + outdir + 'tree_' + str(save_seed) + '.pdf')
    plots.plot_tree(best[0],save_seed,outdir)

if Fitness == True:
    outfile = "fitness-" + file + "-" + evaluate + "-" + str(rseed) + ".pdf"
    print('saving fitness plot to', outfile)
    plots.plot_fitness(fitness,outfile)

