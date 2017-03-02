#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'other',
                 'poi_to_this_person_ratio', 'this_person_to_poi_ratio'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data = pd.DataFrame.from_dict(data_dict, orient='index')

data.replace('NaN',np.nan, inplace=True)
# Drop email-address
data = data.drop(['email_address'], 1)
# Replace null with 0
data.fillna(0, inplace = True)

### Task 2: Remove outliers
data = data.drop(['TOTAL'])
# By reading the provided pdf, drop two more rows
data = data.drop(['THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'])

### Task 3: Create new feature(s)
# Add new features
data['poi_to_this_person_ratio'] = np.where(data['to_messages'] > 0, data['from_poi_to_this_person']/data['to_messages'], data['to_messages'])
data['this_person_to_poi_ratio'] = np.where(data['from_messages'] > 0, data['from_this_person_to_poi']/data['from_messages'], data['from_messages'])

# From dataframe to dict
data_dict = data.to_dict(orient='index')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=42)
kbest = SelectKBest()
scaler = MinMaxScaler()
classifier = DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

pipeline = Pipeline(steps=[('minmax_scaler', scaler), ('feature_selection', kbest), ('decision_tree', classifier)])

# Values of params for tuning
# parameters = dict(decision_tree__min_samples_leaf=range(1, 5),
#                           decision_tree__max_depth=range(1, 10),
#                           decision_tree__class_weight=['balanced'],
#                           decision_tree__criterion=['gini', 'entropy'],
#                           decision_tree__random_state=[42],
#                           feature_selection__k=range(8,18))

# Params of best estimator
# Build pipeline with the best params
parameters = dict(decision_tree__min_samples_leaf=[4],
                  decision_tree__max_depth=[2],
                  decision_tree__class_weight=['balanced'],
                  decision_tree__criterion=['gini'],
                  decision_tree__random_state=[42],
                  feature_selection__k=[16])

# Use GridSearchCV and f1 scores to find the best params
# f1 takes both precision and recall into account
cv_dt = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=sss)
cv_dt.fit(features, labels)

clf = cv_dt.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
