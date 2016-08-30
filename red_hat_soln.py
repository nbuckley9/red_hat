# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 10:44:46 2016

@author: v-nibuck
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#the following simple leak solution was taken and modified from Kevin Palm's Kaggle profile:
#https://www.kaggle.com/kevinpalm/predicting-red-hat-business-value/simplified-leak-starter-template
def simple_load():

    # Read in the data
    people = pd.read_csv("people.csv")
    train = pd.read_csv("act_train.csv")
    test = pd.read_csv("act_test.csv")

    # Merge people to the other data sets
    train = train.merge(people, on="people_id", suffixes=("_act", ""))
    test = test.merge(people, on="people_id", suffixes=("_act", ""))

    # Set index to activity id
    train = train.set_index("activity_id")
    test = test.set_index("activity_id")

    # Correct some data types
    for field in ["date_act", "date"]:
        train[field] = pd.to_datetime(train[field])
        test[field] = pd.to_datetime(test[field])

    #look at merged data train for example    
    print train.head()
    print test.head()
    return train, test


def group_decision(train, test, only_certain=True):
    # Exploit the leak revealed by Loiso and team to try and directly infer any labels that can be inferred
    # https://www.kaggle.com/c/predicting-red-hat-business-value/forums/t/22807/0-987-kernel-now-available-seems-like-leakage

    # Make a lookup dataframe, and copy those in first since we can be sure of them
    lookup = train.groupby(["group_1", "date_act"], as_index=False)["outcome"].mean()
    test = pd.merge(test.reset_index(), lookup, how="left", on=["group_1", "date_act"]).set_index("activity_id")
    # Create some date filling columns that we'll use after we append
    train["date_act_fillfw"] = train["date_act"]
    train["date_act_fillbw"] = train["date_act"]

    # Create some group filling columns for later use
    train["group_fillfw"] = train["group_1"]
    train["group_fillbw"] = train["group_1"]

    # Put the two data sets together and sort
    df = train.append(test)
    df = df.sort_values(by=["group_1", "date_act"])

    # Fill the dates
    df["date_act_fillfw"] = df["date_act_fillfw"].fillna(method="ffill")
    df["date_act_fillbw"] = df["date_act_fillbw"].fillna(method="bfill")

    # Fill labels
    df["outcome_fillfw"] = df["outcome"].fillna(method="ffill")
    df["outcome_fillbw"] = df["outcome"].fillna(method="bfill")

    # Fill the groups
    df["group_fillfw"] = df["group_fillfw"].fillna(method="ffill")
    df["group_fillbw"] = df["group_fillbw"].fillna(method="bfill")

    # Create int booleans for whether the fillers are from the same date
    df["fw_same_date"] = (df["date_act_fillfw"] == df["date_act"]).astype(int)
    df["bw_same_date"] = (df["date_act_fillbw"] == df["date_act"]).astype(int)

    # Create int booleans for whether the fillers are in the same group
    df["fw_same_group"] = (df["group_fillfw"] == df["group_1"]).astype(int)
    df["bw_same_group"] = (df["group_fillbw"] == df["group_1"]).astype(int)

    # Use the filled labels only if the labels were from the same group, unless we're at the end of the group
    df["interfill"] = (df["outcome_fillfw"] *
                       df["fw_same_group"] +
                       df["outcome_fillbw"] *
                       df["bw_same_group"]) / (df["fw_same_group"] +
                                               df["bw_same_group"])

    # If the labels are at the end of the group, cushion by 0.5
    df["needs cushion"] = (df["fw_same_group"] * df["bw_same_group"] - 1).abs()
    df["cushion"] = df["needs cushion"] * df["interfill"] * -0.1 + df["needs cushion"] * 0.05
    df["interfill"] = df["interfill"] + df["cushion"]

    # Fill everything
    df["outcome"] = df["outcome"].fillna(df["interfill"])

    if only_certain == True:
        # Drop anything we're not 100% certain of
        df = df[(df["outcome"] == 0.0) | (df["outcome"] == 1.0)]

    # Return outcomes to the original index
    test["outcome"] = df["outcome"]
    return test["outcome"]


def benchmark_model():

    # Load in the data set simply by merging together
    train, test = simple_load()
  

    # Try to just infer the correct dates using the data leak
    test["outcome"] = group_decision(train, test, only_certain=False)
    
     # Fill any missing rows with using standard classification
    test=missing_value_model(train,test)
    print "THIS IS THE FINAL FILE", test.iloc[220:230,52:]    
    # Write the inferred predictions to a template
    #test.reset_index()[["activity_id", "outcome"]].to_csv("starter_template.csv", index=False)

   
    
    return test.reset_index()[["activity_id", "outcome"]]

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

def missing_value_model(train,test):
   
    
    # Subset train dataframe randomly 
    
    #train=train.sample(frac=0.05, replace=False)    
    test_missing_set=test[test['outcome'].isnull()]
    
    print "test_missing_set has {} samples with {} features each.".format(*test_missing_set.shape)
    print "test_missing_set columns:\n{}".format(test_missing_set.columns.values)
     #drop the extra columns created prior
    test_missing_set=test_missing_set.drop(['outcome'],axis=1)
    
    #drop the additional artificially created columns
    train=train.drop(['date_act_fillfw', 'date_act_fillbw', 'group_fillfw','group_fillbw'],axis=1)
    
    #drop other columns 
    train=train.drop(['people_id','group_1','date','date_act'],axis=1)
    test_missing_set=test_missing_set.drop(['people_id','group_1','date','date_act'],axis=1)
    
    print "Train dataset has {} samples with {} features each.".format(*train.shape)    
    print "Train columns:\n{}".format(train.columns.values)
    
    #encode labels
    train=MultiColumnLabelEncoder().fit_transform(train)
    test_missing_set=MultiColumnLabelEncoder().fit_transform(test_missing_set) 
    
    #split into training and testing sets
    feature_cols=list(train.columns[:].drop(['outcome']))
    target_col=train.outcome
  
    X_all= train[feature_cols]
    y_all= target_col
    
    #create baseline
    counts=train["outcome"].value_counts()
    #baseline percentage assumes we assign the most frequently occurring value, in this case '0'
    baseline_percentage=1.0*counts[0]/(counts[0]+counts[1])
    print "This is the baseline percentage: ", baseline_percentage
    
    # So effectively 55% of the outcomes are '0' and only 45% are '1', so any ML algorithm that beats this is helpful
    #Train Test Split
  


    #fit a random forest classifier
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()#before had n_estimators=100,max_features=20
    clf.fit(X_all,y_all)
    
    test_missing_set['outcome']=clf.predict(test_missing_set)
    #Left join the new test_missing_set to test
    test_missing_set=test_missing_set.reset_index()

    
    print "Test missing dataset has {} samples with {} features each.".format(*test_missing_set.shape)
    test=test.reset_index()
    
    print "Test  has {} samples with {} features each.".format(*test.shape)
    
    #subset test_missing_set to only the values of interest (activity_id and outcome)
    test_missing_set=test_missing_set[['activity_id','outcome']]
    
    #merge those to the original test dataset
    test = pd.merge(test, test_missing_set, how="left", on='activity_id',suffixes=('','_new'))
    print "This is test file before filling NAs", test.iloc[220:230,52:]
    #fill NAs with blanks and then combine the two columns into a final outcome column
    test['outcome']=test[['outcome']].fillna(float(0.0))
    test['outcome_new']=test[['outcome_new']].fillna(float(0.0))
    print 'THESE ARE THE DATATYPES', test['outcome'].dtype, test['outcome_new'].dtype
    test['outcome']=pd.to_numeric(test['outcome'])+pd.to_numeric(test['outcome_new'])
    print "This is the test file after filling NAs", test.iloc[220:230,52:]    
        
    #test["outcome"] = test["outcome"].fillna(test["outcome"].mean())    
    return test

def main():

    # Write a benchmark file to the submissions folder
    benchmark_model().to_csv("benchmark_submission2.csv", index=False)

if __name__ == "__main__":
    main()