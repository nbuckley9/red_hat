# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:00:44 2016

@author: v-nibuck
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

try:
    people_df=pd.read_csv("people.csv")
    act_train_df=pd.read_csv("act_train.csv")
    print "People dataset has {} samples with {} features each.".format(*people_df.shape)
    print people_df.head()
    print "Actions train dataset has {} samples with {} features each.".format(*act_train_df.shape)
    print act_train_df.head()
except:
    print "Dataset(s) could not be loaded. Is the dataset missing?"

#Inner join the two dataframes
merged_df = pd.merge(left=people_df,right=act_train_df, left_on='people_id', right_on='people_id')

#Check that inner join still has complete actions list (2197291 rows, and 15+41-1 features)
print merged_df.shape
print merged_df.head()

#drop the columns that seem to have no bearing on random forest, such as date, id columns
merged_df=merged_df.drop(['people_id','group_1','activity_id','date_y'],axis=1)


# Subset dataframe randomly 
merged_df=merged_df.sample(frac=0.01, replace=False)
print "Sample merged dataset has {} samples with {} features each.".format(*merged_df.shape)
print merged_df.head()

#preprocessing for random forest
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# Create some toy data in a Pandas dataframe
#MultiColumnEncoder adapted from: http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
#Another option here is to use: df.apply(LabelEncoder().fit_transform)
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

merged_df=MultiColumnLabelEncoder().fit_transform(merged_df)

#Updated df with categories encoded
print merged_df.head()

#split into training and testing sets
feature_cols=list(merged_df.columns[:-1])
target_col=merged_df.columns[-1]
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)
X_all= merged_df[feature_cols]
y_all=merged_df[target_col]

#create baseline
counts=merged_df["outcome"].value_counts()
#baseline percentage assumes we assign the most frequently occurring value, in this case '0'
baseline_percentage=1.0*counts[0]/(counts[0]+counts[1])
print "This is the baseline percentage: ", baseline_percentage

# So effectively 55% of the outcomes are '0' and only 45% are '1', so any ML algorithm that beats this is helpful





#EDA on the columns to understand what matters
'''
for col in list(merged_df.columns[13:25]):
    print col
    print merged_df[col].value_counts()
'''

#Preprocess feature columns

    

#creates memory error currently, must drop the non-categorical columns from dummy type

'''
def preprocess_features(X):
#Preprocesses the student data and converts non-numeric binary variables into
        #binary (0/1) variables. Converts categorical variables into dummy variables. 
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['True', 'False'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
       
    return output

X_all = preprocess_features(X_all)

print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
'''

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,stratify=y_all, test_size=0.25, random_state=42)





#fit a random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,max_features=20)
clf.fit(X_train,y_train)

pred=clf.predict(X_test)


#ATTEMPT TO IMPLEMENT GRIDSEARCHCV

from sklearn import metrics
# testing score
#score = metrics.f1_score(y_test, pred, pos_label=list(set(y_test)))
pscore = metrics.accuracy_score(y_test, pred)
print "Random forest score: ", pscore

from sklearn.metrics import make_scorer
from sklearn import grid_search
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = metrics.accuracy_score(y_true,y_predict)
    
    
    # Return the score
    return score
def fit_model(X,y,clf,params):
    #Perform grid search over parameters given and returns optimal param set

    
    #clf=RandomForestClassifier()
    #params = {'n_estimators':[10,20,30],'max_depth':[None],'max_features':[19,20,21]}
    scoring_fnc=make_scorer(performance_metric)

# TODO: Create the grid search object
    grid = grid_search.GridSearchCV(clf,param_grid=params,scoring=scoring_fnc)

# Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X,y)

# Print the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train,RandomForestClassifier(),{'n_estimators':[10],'max_depth':[None],'max_features':[21]})

# Produce the value for 'max_depth'
print "Parameter 'n_estimators',is {} for the optimal model.".format(reg.get_params()['n_estimators'])
print "Parameter 'max_depth', is {} for the optimal model.".format(reg.get_params()['max_depth'])
print "Parameter 'max_features' is {} for the optimal model.".format(reg.get_params()['max_features'])
#fit a naive bayes classifier
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)

pred=clf.predict(X_test)

from sklearn import metrics
# testing score
#score = metrics.f1_score(y_test, pred, pos_label=list(set(y_test)))
pscore = metrics.accuracy_score(y_test, pred)
print "Naive bayes score: ", pscore
#fit a KNN classifier

#fit KNN

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
pscore = metrics.accuracy_score(y_test, pred)

print "KNN score: " , pscore

#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
pscore = metrics.accuracy_score(y_test, pred)

print "DecisionTree score: " , pscore           


#Gradient boosting

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.15, max_depth=11, random_state=0).fit(X_train, y_train)
pred=clf.predict(X_test)
pscore=metrics.accuracy_score(y_test,pred)

print "Gradient Boosting score: " , pscore           

#reg = fit_model(X_train, y_train,GradientBoostingClassifier(),{'learning_rate':[0.15],'n_estimators':[50,70,80,100],'max_depth':[9]})

# Produce the value for 'max_depth'
#print "Parameter 'n_estimators',is {} for the optimal model.".format(reg.get_params()['n_estimators'])
#print "Parameter 'max_depth', is {} for the optimal model.".format(reg.get_params()['max_depth'])
#print "Parameter 'learning_rate' is {} for the optimal model.".format(reg.get_params()['learning_rate'])
'''
#Adaboost classifier
#This works but we're not using it because it take a lot longer to run
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None), n_estimators=50)
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
pscore = metrics.accuracy_score(y_test, pred)

print "Adaboost w/DecisionTree score: " , pscore           


#xgboost classifer
#need to install xgboost to get this working
import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
pred = gbm.predict(X_test)
pscore = metrics.accuracy_score(y_test, pred)

print "xgboost score: " , pscore 

#fit SVM classifier
#SVM times out so saving it for later

from sklearn import svm
clf=svm.SVC()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
pscore = metrics.accuracy_score(y_test, pred)

print "SVM score: " , pscore
'''













    
