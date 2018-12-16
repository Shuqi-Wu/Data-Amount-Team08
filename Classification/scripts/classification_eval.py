'''
Created on Oct 19, 2018

@author: Geese Howard

'''
import os
import math
import pandas
import argparse
import warnings
import numpy as np
import matplotlib.pylab as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
DEFAULT_INTERVALS = 1000
SCRIPT_DIR = os.path.abspath(os.path.join(__file__,os.pardir))
OUTPUT_DIR = os.path.join(SCRIPT_DIR,'output')

def plot(x,y,x_label='X-Axis',y_label='Y-Axis'):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x,y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return (fig,ax)

def save_plot(fig,filename,save_to_dir=OUTPUT_DIR):
    output_filepath = os.path.join(save_to_dir,filename)
    fig.savefig('%s.png'%output_filepath)

def main(dataset_path,intervals,cv_scheme,cols_to_normalize=None):

    outfolder_name = os.path.basename(os.path.abspath(os.path.join(dataset_path,
                                                                   os.pardir)))
    ds_output = os.path.join(OUTPUT_DIR,outfolder_name)
    if not os.path.exists(ds_output):
        os.makedirs(ds_output)

    df = pandas.read_csv(dataset_path)
    array = df.values.astype(np.float)
    num_data_pts = len(array)
    step = int(num_data_pts/intervals)
    train_sizes = list(range(step,num_data_pts,step))
    X=array[:,:-1]

    if cols_to_normalize is not None:
        min_max_scaler = preprocessing.MinMaxScaler()
        for col_number in cols_to_normalize:
            col_unscaled = X[:,col_number]
            col_scaled = min_max_scaler.fit_transform(col_unscaled.reshape(-1,1))
            X[:,col_number] = col_scaled.reshape(1,-1)

    y=array[:,-1]
    scoring = {'acc': 'accuracy',
               'n_loss':'neg_log_loss',
               'precision':'precision',
               'recall':'recall',
               'roc_auc':'roc_auc'}

    accuracy = []
    neg_log_loss = []
    precision = []
    recall = []
    roc_auc = []
    c_intervals = []
    train_sizes_X = []
    test_sizes_X = []
    z_95 = 1.96
    for train_size in train_sizes:
        model = LogisticRegression()
        test_size = num_data_pts - train_size
        #K-Fold disabled for now
        if cv_scheme == 0:
            cv_arg = model_selection.ShuffleSplit(n_splits=2,
                        train_size=train_size, test_size=test_size,
                        random_state=0)
        else:
            cv_arg = int(num_data_pts/train_size)
            if cv_arg<4:
                break

        scores = model_selection.cross_validate(model, X, y, scoring=scoring,
                                 cv=cv_arg, return_train_score=False)

        train_sizes_X.append(train_size)
        test_sizes_X.append(test_size)
        neg_log_loss.append(scores['test_n_loss'].mean())
        
        acc = scores['test_acc'].mean()
        error = 1 - acc
        c_interval_len = 2 * z_95 * math.sqrt( (error * (1 - error)) / test_size)
        accuracy.append(acc)
        c_intervals.append(c_interval_len)

        precision.append(scores['test_precision'].mean())
        roc_auc.append(scores['test_roc_auc'].mean())
        recall.append(scores['test_recall'].mean())

    train_label = 'Number of Training samples'
    test_label = 'Number of Testing samples'
    p_acc = plot(train_sizes_X[:780],accuracy[:780],x_label=train_label,y_label='Accuracy')
    p_lloss = plot(train_sizes_X[:780],neg_log_loss[:780],x_label=train_label,y_label='Negative Log Loss')
    p_precision = plot(train_sizes_X[:780],precision[:780],x_label=train_label,y_label='Precision')
    p_roc_auc = plot(train_sizes_X[:780],roc_auc[:780],x_label=train_label,y_label='ROC Area Under curve')
    p_recall = plot(train_sizes_X[:780],recall[:780],x_label=train_label,y_label='Recall')
    p_c_inverval = plot(test_sizes_X[:780],c_intervals[:780],x_label=test_label,y_label='Length of 95% Confidence interval')

    save_plot(p_acc[0],'accuracy',ds_output)
    save_plot(p_lloss[0],'log_loss',ds_output)
    save_plot(p_precision[0],'precision',ds_output)
    save_plot(p_roc_auc[0],'roc_auc',ds_output)
    save_plot(p_recall[0],'recall',ds_output)
    save_plot(p_c_inverval[0],'c_intervals',ds_output)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Argument parser of classification'\
                                                'evaluation script')
    parser.add_argument('--dset_path', type=str,help='Absolute path to the FORMATTED dataset')
    parser.add_argument('--intervals', type=int,help='How many intervals to divide the dataset into')
    parser.add_argument('--xy_split', action='store_true')
    parser.add_argument('--cols_to_normalize', type=str,help='Comma separated list of'\
                        ' column numbers to normalize to -1 to 1 range')
    parser.add_argument('--kfolds', action='store_true')

    args = parser.parse_args()

    if not args.dset_path:
        raise Exception('Please provide the Dataset path using the --dset_path switch')
    else:
        if not os.path.exists(args.dset_path):
            raise Exception('Dataset does not exist on the specified path')

    if not args.intervals:
        args.intervals = DEFAULT_INTERVALS
    else:
        if args.intervals < 4:
            raise Exception('Intervals cannot be less than 4')

    if args.cols_to_normalize:
        args.cols_to_normalize = [int(x) for x in args.cols_to_normalize.split(',')]

    if args.xy_split and args.kfolds:
        raise Exception('Only a single cross validation scheme can be specified ata  time')
    elif not args.xy_split and not args.kfolds:
        cv_scheme = 0
    elif args.xy_split:
        cv_scheme = 0
    elif args.kfolds:
        cv_scheme = 1
    else:
        #All 4 cases have been covered
        pass

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    main(args.dset_path,args.intervals,cv_scheme,cols_to_normalize=args.cols_to_normalize)
