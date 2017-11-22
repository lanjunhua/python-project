# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:36:40 2017

@author: junhua.lan
"""



#This file is contian the feature engineering 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


#str = unicode(str, errors='ignore')

# get the data
train = pd.read_csv('D:\0.accumulate\1.july onine\kaggle\20161224第一课\第1课\Feature_engineering_and_model_tuning\Kaggle_Titanic\train.csv')
train = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\test\\Train.csv')
test = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\test\\test.csv')



train1 = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\Train.csv')
test1 = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\test1\\test.csv')


train2 = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\test1\\train_modified.csv')
test2 = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\test1\\test_modified.csv')



print(train.shape, test.shape)
print(train.dtypes)
train.head(5)

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index = True)
data.shape


# think the NA Value
data.apply(lambda x: sum(x.isnull()))

#dfTest = data.apply(lambda x: x.value_counts())



vars = ['Pclass', 'Sex', 'Age', 'Parch', 'SibSp','Ticket', 'Cabin', 'Embarked']
for var in vars:
    print('\n%s the different values counts\n' %var)
    print(data[var].value_counts())
    
    
len(data['Name'].unique())
data.drop('Name', axis = 1, inplace = True)


data['Age'] = data['Bob'].apply(lambda x: 115 - int(x[-2:]))
data.drop('Bob', axis = 1, inplace = True)

data.boxplot(column = ['Fare'], return_type = 'axes')

data['Age_Miss'] = data['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
data[['Age', 'Age_Miss']].head(10)
data['Age'].describe()
data['Age'].fillna(29, inplace = True)

data['Pclass'].value_counts()
data['Embarked'].value_counts()



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Embarked']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])

data.dtypes


data = pd.get_dummies(data, columns = var_to_encode)




# xgboost model training
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
        
        
train = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\test1\\train_modified.csv')
test = pd.read_csv('C:\\Users\\junhua.lan\\Desktop\\test1\\test_modified.csv')

train.shape
test.shape
target = 'Disbursed'
IDcol = 'ID'

train['Disbursed'].value_counts()
# get the model and crossvalidation
# 1.get the model
# 2.get the trainging data's precision
# 3.get the auc of the training
# 4.use xgboost crossvalidation and update the n_estimators
# 5.get the picture of the important variables


def modelfit(alg, dtrain, dtest, predictors, useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain[target].values)
        xgtest  = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'],
                          nfold = cv_folds, early_stopping_rounds = early_stopping_rounds)
        
        alg.set_params(n_estimators = cvresult.shape[0])
        
        # training model
        alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric = 'auc')
        
        
        # predict the training data
        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
        
        # return the model result
        print('accuracy: %.4g' % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
        print('auc score of training: %f' % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
        
        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending = False)
        feat_imp.plot(kind = 'bar', title = 'Feature Importance')
        plt.ylabel('Feature Importance Score')
        
        
        
# step1: get the model
predictors = [x for x in train.columns if x not in [target, IDcol]]

xgb1 = XGBClassifier(
        learning_rate = 0.1,
        n_estimators = 1000,
        max_depth = 5,
        min_child_weight = 1,
        gamma = 0,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary:logistic',
        nthread = 4,
        scale_pos_weight = 1,
        seed = 27)

modelfit(xgb1, train, test, predictors)



# step2: get the good params
param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
        }



param_test1 = {
        'max_depth': [3,5,7,9],
        'min_child_weight': [1,3,5,6]
        }

gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
                                                  n_estimators = 140,
                                                  max_depth = 5,
                                                  min_child_weight = 1, 
                                                  gamma = 0, 
                                                  subsample = 0.8,
                                                  colsample_bytree = 0.8,
                                                  objective = 'binary:logistic',
                                                  nthread = 4,
                                                  scale_pos_weight = 1,
                                                  seed = 27),param_grid = param_test1, scoring = 'roc_auc',n_jobs = 4, iid = False, cv = 5)




gsearch1.fit(train[predictors], train[target])

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
