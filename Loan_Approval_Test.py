# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Quantifi Lending Club Test
#
# In this notebook you will build out a simple data science workflow to solve a predictive problem.
#
# We have included a dataset, 10K_Lending_Club_Loans.csv. This dataset comes courtesy of Lending Club, an online peer-to-peer lending website that matches those looking to borrow money with those with a little extra to lend out. Lending Club includes a number of descriptive features as well as a target (is_bad) indicating whether the loan was paid back successfully. 
#
# Your task is to:
# * Perform a simple exploratory data analysis
# * Train a model to predict **is_bad**
# * Evaluate accuracy
# * **Answer questions about your model**
#
# There is no right or wrong answer, this test is all about showcasing your specific style.
#
# First, the dataset:

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("./10K_Lending_Club_Loans.csv")

df.head(5)

df.columns

df.info()

# ## Perform exploratory data analysis here:
# * Explore a few features visually
# * Explore the target output

from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Pandas Profiling Report")

profile

# ## Clean dataset here
# * Remove columns, transform variables, etc

df.head(1)

# underballanced terms
df['term'].value_counts()

# remove % and convert to float
df['int_rate'] = df['int_rate'].map(lambda x: float(x.rstrip('%')))

#underbalanced grades
df['grade'].value_counts()

# +
# lots of strings here as a next step 
# i would work on this list to see if they are belonging to some groups or duplicated
# i will drop this column for now

df['emp_title'].value_counts()
# -

df = df.drop(columns=['emp_title'])

# we've got 259 records with nans I will drop them
df['emp_length'].value_counts().sum()

df = df.dropna(subset = ['emp_length'])

# 1 NONE ownership ==drop
df['home_ownership'].value_counts()
df = df[df['home_ownership']!='NONE']

df['annual_inc'].value_counts()

df.shape

df['verification_status'].value_counts()

# this column is not informative == drop
df['pymnt_plan'].value_counts()

df = df.drop(columns = ['pymnt_plan'])

df.shape

# url is not informative - drop
df = df.drop(columns = ['url'])

# desc is not informative now. but i would drill in to see some key words and patterns here. Will delete for now
df = df.drop(columns = ['desc'])


# purpose - OK
df['purpose'].value_counts()

# title is not informative now. but i would drill in to see some key words and patterns here. Will delete for now
df['title'].value_counts()
df = df.drop(columns = ['title'])

#  zip code is potentially usefull as we can get some info on the grouping of the loans and so on.
# I will remove for now as it has too many categories
df['zip_code'].value_counts()
df = df.drop(columns = ['zip_code'])

# addr_state - OK
df['addr_state'].value_counts()

# dti - OK
df['dti'].value_counts().sum()

# might be usefuil in the future, as we get more data but non informative for now - drop
df['delinq_2yrs'].value_counts()
df = df.drop(columns=['delinq_2yrs'])

#  i would calculate the 'tenure' of the client between today and the date
df['earliest_cr_line'].value_counts()
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x:pd.to_datetime(x))

df['earliest_cr_line_tenure'] = df['earliest_cr_line'].apply(lambda x: (pd.to_datetime("today")- x) / np.timedelta64(1, 'D'))

df['earliest_cr_line_tenure'].head()

# eliminating data with datetime errors
df = df[df['earliest_cr_line_tenure']>0]

df = df.drop(columns=['earliest_cr_line'])

df.head(1)

# inq_last_6mths - dropped nans
df['inq_last_6mths'] = df['inq_last_6mths'].dropna()
df['inq_last_6mths'].value_counts().sum()

df['mths_since_last_delinq'].unique()

df['mths_since_last_delinq'] = df['mths_since_last_delinq'].dropna()
df['mths_since_last_delinq'].value_counts().sum()

# due to the lack of time for this assignment i will drop this column as it is not informative at the moment
# but in future i would improvise with SMOTE in here
df['mths_since_last_delinq'].value_counts()

df = df.drop(columns = ['mths_since_last_delinq'])

df.head(1)

#  mths_since_last_record   non informative - drop
df['mths_since_last_record'].value_counts().sum()

df = df.drop(columns=['mths_since_last_record'])

#  open_acc - OK
df['open_acc'].value_counts().sum()

# non informative - drop
df['pub_rec'].value_counts()

df = df.drop(columns=['pub_rec'])

# revol_bal - OK
df['revol_bal'].value_counts().sum()

# revol_util - OK
df['revol_util'].value_counts().sum()

# total_acc - OK
df['total_acc'].value_counts().sum()

# non informative - drop
df['initial_list_status'].value_counts()

df =df.drop(columns=['initial_list_status'])

df = df.drop(columns = ['mths_since_last_major_derog'])

# non informative
df['policy_code'].value_counts()

df = df.drop(columns=['policy_code'])

df.shape

df = df.dropna()
df.shape

df.head(5)

profile = ProfileReport(df, title="Pandas Profiling Report")

profile

# +
# there is a strong correlation between loan amount and funded ammount
# -

df['loan_amnt'].head(5)

df['funded_amnt'].head(5)

# I will remove funded ammount for now as it is repeating loan amount
df = df.drop(columns=['funded_amnt'])

# also iinstallemnt is highly correlated to income and loan amount. 
# I would suggest running a stats test like backpropagation to see how important 
# these features are for the target. I will skip this step for now by deleting installment
df = df.drop(columns=['installment'])

df.info()





# ### Prep data for modeling - downsample

# Very underballanced dataset i will downsample for now to send to the model
# in the future we can try upsampling or SMOTE
df['is_bad'].value_counts()

# +
from sklearn.utils import resample
df_majority = df[df.is_bad==0]
df_minority = df[df.is_bad==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=1237,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.is_bad.value_counts()
# -

# ### Prep data for modeling - onehot encode

df_downsampled.info()

# avoiding multicoliniarity trap we are deleting the firs value in the getdummies
object_columns = df_downsampled.select_dtypes(include=['object']).columns.to_list()
df_for_ml = pd.get_dummies(data=df_downsampled, columns=object_columns, drop_first =True)

df_for_ml.head()

# ## Fit a model here and evaluate the accuracy
# * Use whatever model or library you want

# +
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.model_selection import cross_val_score
from copy import deepcopy
from sklearn.metrics import make_scorer
from typing import Dict
from itertools import product
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging
import pathlib
import sys
import argparse
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from scipy.stats import pearsonr
# -

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# set a dictionary of the grid search parameters
ridge_param_dist = {'alpha': [1e-5, 1e-2, 0.1, 1, 10, 100],
                    'random_state': [33]
                    }

rf_param_dist = {'n_estimators': [10, 20, 100, 500, 1000],
                 'max_depth': [None, 5, 10, 30],
                 'max_features': ['sqrt', 'log2'],
                 'random_state': [33],
                 }

lgb_param_dist = {'n_estimators': [100, 500, 1000],
                  'num_leaves': [8, 15, 24, 35, 48, 63, 80],
                  'subsample_freq': [0, 1, 2, 5, 10],
                  'subsample': [0.5, 0.7, 0.8, 0.9],
                  'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                  'subsample_for_bin': [5, 10, 20, 50, 100],
                  'boosting_type': ['gbdt', 'dart'],
                  'max_depth': [3, 4, 5, 6, 7, 8, 9],
                  'colsample_bytree': [0.5, 0.7, 0.8, 0.9],
                  'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 10, 100],
                  'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 10, 100],
                  'random_state': [33]
                  }

lgb_param_class_dist = {'n_estimators': [100, 500, 1000],
                  'num_leaves': [8, 15, 24, 35, 48, 63, 80],
                  'bagging_freq': [1, 2, 5, 10],
                  'subsample': [0.5, 0.7, 0.8, 0.9],
                  'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                  'subsample_for_bin': [5, 10, 20, 50, 100],
                  'boosting_type': ['gbdt', 'dart','rf'],
                  'max_depth': [3, 4, 5, 6, 7, 8, 9],
                  'colsample_bytree': [0.5, 0.7, 0.8, 0.9],
                  'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 10, 100],
                  'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 10, 100],
                  'class_weight': [None,'balanced'],
                  'bagging_fraction':[0.5],
                  'random_state': [33]
                  }

estimator_params = {LGBMRegressor(): lgb_param_dist,
                    RandomForestRegressor(): rf_param_dist,
                    Ridge(): ridge_param_dist}


estimator_params_class = {RandomForestClassifier(): rf_param_dist,
                    LGBMClassifier(): lgb_param_class_dist,
                    RidgeClassifier(): ridge_param_dist}

# set logger
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger()

# get the path to data folder
from pathlib import Path
__file__ = Path.cwd()


def get_param_list(param_dist: Dict, max_selection: int):
    """
    If the number of parameter combinations <= max_selection, use all parameter
    combinations in the grid. Otherwise use random generated grid to max_selection
    :param param_dist: Hyperparameters in forms of {param_name_1: [list of possible para values],
                                                    param_name_2: [list of possible para values],
                                                    ...}
    :param max_selection: Maximum combinations to randomly select
    """
    max_possible_combinations = len(list(product(*list(param_dist.values()))))
    ans = []
    if max_possible_combinations <= max_selection:  # use all combinations
        keys = []
        vals = []
        for key, val in param_dist.items():
            keys.append(key)
            vals.append(val)
        param_list = product(*vals)
        for param in param_list:
            ans.append({k: v for k, v in list(zip(keys, param))})
    else:  # use random grid
        for _ in range(max_selection):
            ans.append({k: np.random.choice(v) for k, v in param_dist.items()})
    return ans


def cv_suit(estimators, x, y, cv=3, scoring=None, random_state=33):
    """
    For each estimator in estimators, run cross validation. Return the estimator that gives the highest score given
    by mean test_score from cross validation.
    """
    cv_scores = []
    for estimator in estimators:
        scores = cross_val_score(estimator, X=x, y=y, scoring=scoring, cv=cv)
        cv_scores.append(np.mean(scores))
    max_idx = np.argmax(cv_scores)
    return estimators[max_idx]


def main_class(df_orig, train_df, test_df, scoring=mean_squared_error, n_max_param_combinations=30):
    """This function runs gridsearch and selects the best model with best hyper parameters to predict target

    :param use_pca: Specify whether to use PCA
    :param ncomponents: Specify threshold of explained variance
    :param scoring: specify which scorring to use when comparing models
    :param n_max_param_combinations: Specify what would be the upper limit oh hyper parameter combinations to search in
    :param normalize: Specify whether we want to use normalized data frame
    :return: returns predicted target and plots the correlations
    """
    x_train = train_df.drop('is_bad', axis = 1).values.tolist()
    y_train = train_df['is_bad'].values.tolist()
    index = test_df.index
    
    x_test = test_df.drop('is_bad', axis = 1).values.tolist()


    estimators = []
    for estimator in estimator_params_class.keys():
        param_dist = estimator_params_class.get(estimator)
        if param_dist is not None:
            param_list = get_param_list(param_dist, n_max_param_combinations)
            for params in param_list:
                estimators.append(deepcopy(estimator).set_params(**params))
        else:
            estimators.append(estimator)

    scorer = make_scorer(scoring, greater_is_better=False)

    selected_estimator = cv_suit(estimators, x=x_train, y=y_train, scoring=scorer)
    
    logging.info(selected_estimator)
    selected_estimator.fit(x_train, y_train)
    
    
    predicted = selected_estimator.predict(x_test)
    
    predicted = pd.DataFrame(data=np.array(list(zip(predicted))), columns=['predicted'], index=index)
    
    conf_mat = confusion_matrix(y_true=test_df['is_bad'].values.tolist(), y_pred=predicted)
    print('Confusion matrix:\n', conf_mat)

    labels = ['Class 0', 'Class 1']
    n_classes = len(labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()
    print (classification_report(test_df['is_bad'].values.tolist(), predicted, target_names=labels))
    
    return selected_estimator


train_df, test_df = train_test_split(df_for_ml, random_state = 33)

selected_estimator = main_class(df_for_ml, train_df, test_df, scoring=mean_squared_error, n_max_param_combinations=10)

# Weighted f-1 score is .64 which means that we are predicting a bit better than random guess. We have to work on the model

# ## Explore your model and answer these questions
# You can answer using text, code, or both. 

# * **What is the most important feature for your model?**

print(selected_estimator.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(selected_estimator.feature_importances_, index=train_df.drop('is_bad', axis = 1).columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# int_rate seems to be the most important feature. We would have to investigate this result

#

# * **Why did you pick this model?**
# Historically ridge lgbm, xgboost and Random Forest are the go to when dealing with classifications. I picked three and send through the grid search with 10 combinations and cross validation to make a prediction. If we increase # of combinations we can yield better results

#

# * **What were some of the limitations of your approach?**
# Slow processing time and hard to explain the results

#

# * **What would you like to try next?**
# I have already suggested some additional steps:
# 1. emp_title has lots of strings here as a next step i would work on this list to see if they are belonging to some groups or duplicated
# 2. term is underballanced need to work on data collection
# 3. desc is not informative now. but i would drill in to see some key words and patterns here
# 4.  mths_since_last_delinq mneeds improvisation with SMOTE
# 5. zip code is potentially usefull as we can get some info on the grouping of the loans and so on
# 6. installemnt is highly correlated to income and loan amount. I would suggest running backpropagation to see how important these features are for the target
# 7. In terms of modeling there are bunch of approaches we need to try. I can make an MLP that will be learning the relation between the data/playing with learning rate and architecture
# 8. With proposed models: - run more iterations on grid search, run PCA before the modeling, play with balancing and info gain
# 9. Get the info on what is going with feature importances in the selected_estimator.feature_importances_ as they look odd
#

#

# ## Conclusion
# Save this notebook, it will be your submission. Thank you for participating!

# Lots of fun and lots of things to experiment with. THX
