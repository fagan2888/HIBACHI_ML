#!/usr/bin/env python
#==============================================================================
#
#          FILE:  IO.py
# 
#         USAGE:  import IO (from hib.py)
# 
#   DESCRIPTION:  graphing and file i/o routines.  
# 
#       UPDATES:  170213: added subset() function
#                 170214: added getfolds() function
#                 170215: added record shuffle to getfolds() function
#                 170216: added addnoise() function
#                 170217: modified create_file() to name file uniquely
#                 170302: added plot_hist() to plot std
#                 170313: added get_arguments()
#                 170319: added addone()
#                 170329: added np.random.shuffle() to read_file_np() 
#                 170410: added option for case percentage
#                 170420: added option for output directory
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.10
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Thu Apr 20 09:29:11 EDT 2017
#==============================================================================
import pandas as pd
import csv
import numpy as np
import argparse
import sys
import os
###############################################################################
def get_arguments():
    options = dict()

    parser = argparse.ArgumentParser(description = \
        "Run hibachi evaluations on your data")

    parser.add_argument('-e', '--evaluation', type=str,
            help='name of evaluation [normal|folds|subsets|noise]' +
                 ' (default=normal)')
    parser.add_argument('-f', '--file', type=str,
            help='name of training data file (REQ)' +
                 ' filename of random will create all data')
    parser.add_argument("-g", "--generations", type=int, 
            help="number of generations (default=40)")
    parser.add_argument("-i", "--information_gain", type=int, 
            help="information gain 2 way or 3 way (default=2)")
    parser.add_argument('-o', '--outdir', type=str,
            help='name of output directory (default = .)' +
            ' Note: the directory must exist')
    parser.add_argument("-p", "--population", type=int, 
            help="size of population (default=100)")
    parser.add_argument("-r", "--random_data_files", type=int, 
            help="number of random data to use instead of files (default=0)")
    parser.add_argument("-s", "--seed", type=int, 
            help="random seed to use (default=random value 1-1000)")
    parser.add_argument("-R", "--rows", type=int, 
            help="random data rows (default=1000)")
    parser.add_argument("-C", "--columns", type=int, 
            help="random data columns (default=3)")
    parser.add_argument("-S", "--statistics", 
            help="plot statistics",action='store_true')
    parser.add_argument("-T", "--trees", 
            help="plot best individual trees",action='store_true')
    parser.add_argument("-F", "--fitness", 
            help="plot fitness results",action='store_true')
    parser.add_argument("-P", "--percent", type=int,
            help="percentage of case for case/control (default=25)")
    
  ################################################ 
    parser.add_argument("-U", "--up", type=str,
            help="ML Function accuracy to increase (default=Decision Tree)")
    parser.add_argument("-D", "--down", type=str,
            help="ML Function accuracy to decrease (default=Logistic Regression [L1])")
  ################################################ 
    args = parser.parse_args()


  ################################################  
    
    if(args.up == None):
        options['up'] = 'dt'
    else:
        options['up'] = args.up
        
    if(args.down == None):
        options['down'] = 'lr1'
    else:
        options['down'] = args.down
        
  ################################################  

    if(args.file == None):
        print('filename required')
        sys.exit()
    else:
        options['file'] = args.file
        options['basename'] = os.path.basename(args.file)
        options['dir_path'] = os.path.dirname(args.file)

    if(args.outdir == None):
        options['outdir'] = "./"
    else:
        options['outdir'] = args.outdir + '/'

    if(args.seed == None):
        options['seed'] = -999
    else:
        options['seed'] = args.seed

    if(args.percent == None):
        options['percent'] = 25
    else:
        options['percent'] = args.percent
        
    if(args.population == None):
        options['population'] = 100
    else:
        options['population'] = args.population

    if(args.information_gain == None):
        options['information_gain'] = 2
    else:
        options['information_gain'] = args.information_gain

    if(args.random_data_files == None):
        options['random_data_files'] = 0
    else:
        options['random_data_files'] = args.random_data_files

    if(args.generations == None):
        options['generations'] = 40
    else:
        options['generations'] = args.generations

    if(args.evaluation == None):
        options['evaluation'] = 'normal'
    else:
        options['evaluation'] = args.evaluation

    if(args.rows == None):
        options['rows'] = 1000
    else:
        options['rows'] = args.rows

    if(args.columns == None):
        options['columns'] = 3
    else:
        options['columns'] = args.columns

    if(args.statistics):
        options['statistics'] = True
    else:
        options['statistics'] = False

    if(args.trees):
        options['trees'] = True
    else:
        options['trees'] = False

    if(args.fitness):
        options['fitness'] = True
    else:
        options['fitness'] = False

    return options
###############################################################################
def get_random_data(rows, cols, seed=None):
    """ return randomly generated data is shape passed in """
    if seed != None: np.random.seed(seed)
    data = np.random.randint(0,3,size=(rows,cols))
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def create_file(x,result,distribution,score,outfile):
    d = np.array(x).transpose()    
    columns = [0]*np.shape(d)[1]
    for i in range(0,np.shape(d)[1]): # create columns names for variable number of columns.
        columns[i] = 'X' + str(i)
#    df = pd.DataFrame(np.array(x).transpose(), columns=['X0','X1','X2'])
    df = pd.DataFrame(d, columns=columns)
    df['Class'] = result
    df['Distribution'] = distribution
    df1 = pd.DataFrame()
    df1['Fitness'] = score
    df2 = pd.concat([df,df1], axis=1)
    df2.to_csv(outfile, sep='\t', index=False)
###############################################################################
def read_file(fname):
    """ return both data and x
        data = rows of instances
        x is data transposed to rows of features """
    data = np.genfromtxt(fname, dtype=np.int, delimiter='\t') 
    np.random.shuffle(data) # give the data a good row shuffle
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def printf(format, *args):
    """ works just like the C/C++ printf function """
    import sys
    sys.stdout.write(format % args)
    sys.stdout.flush()
