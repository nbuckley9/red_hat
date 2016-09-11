# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 10:44:46 2016

@author: v-nibuck
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from IPython.display import display
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import renders as rs
from sklearn.ensemble import GradientBoostingClassifier

#the following simple leak solution was taken and modified from Kevin Palm's Kaggle profile:
#https://www.kaggle.com/kevinpalm/predicting-red-hat-business-value/simplified-leak-starter-template
def simple_load():

    # Read in the data
    people = pd.read_csv("people.csv",na_values='')
    train = pd.read_csv("act_train.csv",na_values='')
    test = pd.read_csv("act_test.csv",na_values='')

    # Merge people to the other data sets
    train = train.merge(people, on="people_id", suffixes=("_act", ""))
    test = test.merge(people, on="people_id", suffixes=("_act", ""))

    # Set index to activity id
    train = train.set_index("activity_id")
    test = test.set_index("activity_id")
    
     #Create baseline
    counts=train["outcome"].value_counts()
    #baseline percentage assumes we assign the most frequently occurring value, in this case '0'
    baseline_percentage=1.0*counts[0]/(counts[0]+counts[1])
    print "This is the baseline percentage: ", baseline_percentage
    # So effectively 55% of the outcomes are '0' and only 45% are '1', so any ML algorithm that beats this is helpful

    # Correct some data types
    for field in ["date_act", "date"]:
        train[field] = pd.to_datetime(train[field])
        test[field] = pd.to_datetime(test[field])
    
    #include week number and day number in each
    train["week"] = train["date"].dt.weekofyear
    train["day"] = train["date"].dt.dayofweek
    test["week"] = test["date"].dt.weekofyear
    test["day"] = test["date"].dt.dayofweek
    
    #look at merged data train for example    
    print "INITIAL TRAIN DATA\n", train.head(), train.columns.values
    print "INITIAL TEST  DATA\n", test.head()
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
    #test=missing_value_model_2(train,test)
    #print "THIS IS THE FINAL FILE", test.iloc[220:230,52:]    
    # Write the inferred predictions to a template
    #test.reset_index()[["activity_id", "outcome"]].to_csv("starter_template.csv", index=False)
    
    # Fill any missing rows with the mean of the whole column
    test["outcome"] = test["outcome"].fillna(missing_value_model_4(train,test))
   
    
    return test.reset_index()[["activity_id", "outcome"]]

#MultiColumnEncoder adapted from: http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
#Another option here is to use: df.apply(LabelEncoder().fit_transform)
#ThisLabelEncoder is designed to help preprocess the categorical data into numerical data that can be used by RandomForest Classifier

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
        
#missing value model 4 uses a PCA model to reduce the dimensions before making a prediction 
def missing_value_model_4(train,test):
   
    print "These are train the columns at the beginning of MM4", train.columns.values
    print "These are test the columns at the beginning of MM4", test.columns.values

    #Take all columns and remove the ones created for the leak soln
    train=train.drop(['date_act_fillfw', 'date_act_fillbw', 'group_fillfw','group_fillbw'],axis=1)
    
    #drop additional columns that we can't use easily in normalizer or PCA
    train=train.drop(['date_act','date','people_id'],axis=1)
    test=test.drop(['date_act','date','people_id'],axis=1)
    #TODO take the dtypes that are not already encoded as integers 
    #create a list of these columns to convert using multicolumnlabelencoder
    #for now dropping the columns that seem to have floats or other issues that could cause probs
    
    
    X_all=train.drop(['outcome'],axis=1)
    y_all= train['outcome']
    
    test_x=test  
    
    #preprocess the data using multi-column label encoder
    #TODO only apply to select columns that are non-int,non-float
    X_all=X_all.drop(['char_38'],axis=1)
    X_all=MultiColumnLabelEncoder().fit_transform(X_all)
    X_all=X_all.append(train[['char_38']])
    
    test_x=test_x.drop(['char_38'],axis=1)
    test_x=MultiColumnLabelEncoder().fit_transform(test_x)
    test_x=test_x.append(test[['char_38']])
    
    #impute missing values, normalize and use PCA to pivot to a smaller set of values, then apply gradient boosting classifier
    
    #first two PCA dimensions cover 90%+ of variance so we'll look at those
    
    clf=Pipeline([("Imputer",Imputer(missing_values="NaN",strategy="most_frequent",axis=0)),
                    ("Normalizer",Normalizer()), 
                    ("PCA",PCA(n_components=4)),
                    ("Gradient_boost",GradientBoostingClassifier())])
    
    #clf = GradientBoostingClassifier()#before had n_estimators=100,max_features=20
    clf.fit(X_all,y_all)
    #pca_results=rs.pca_results(X_all,PCA)
    #print "PCA Explained Variance", pca_results['Explained Variance'].cumsum()
    #predict the new outcome variale
   
    transformed_data=clf.predict(test_x)
    test['outcome']=clf.inverse_transform(transformed_data)
    
    return test['outcome']
    
#missing value model 3 uses a random forest model on the columns showing most variation in the PCA model
def missing_value_model_3(train,test):
   
    
    col_list_train=['char_10_act', 'char_1', 
    'char_2', 'char_3', 'char_4', 'char_5', 
    'char_6', 'char_7', 'char_8','char_9', 
    'char_38']
    
    col_list_test=['char_10_act', 'char_1', 
    'char_2', 'char_3', 'char_4', 'char_5', 
    'char_6', 'char_7', 'char_8','char_9', 
    'char_38']
    
    

    X_all= train[col_list_train]
    print "THIS IS X ALL HEAD \n", X_all.head()
    y_all= train['outcome']
    print "THIS IS Y ALL HEAD \n",  y_all.head()
    test_x=test[col_list_test]    
    
  
    
    #encode the data, impute missing values, normalize and fit a gradient boosting classifier
    
    clf=Pipeline([("Label_Encoder",MultiColumnLabelEncoder()),
                  ("Imputer",Imputer(missing_values="NaN",strategy="most_frequent",axis=0)),
                  ("Normalizer",Normalizer()),
                  ("Gradient_boost",GradientBoostingClassifier())])
    
    #clf = GradientBoostingClassifier()#before had n_estimators=100,max_features=20
    clf.fit(X_all,y_all)
    
    #predict the new outcome variale
    test['outcome']=clf.predict(test_x)
    
    
    return test['outcome']
def missing_value_model_2(train,test):
   
    
    col_list_train=[]
    for col in train.columns.values:
        try:
            train[col]=train[col].astype(bool)
            col_list_train.append(col)
        except: 
            pass
    col_list_test=[]
    for col in  test.columns.values:
        try:
            test[col]=test[col].astype(bool)
            col_list_test.append(col)
        except: 
            pass
    

    X_all= train[col_list_train]
    print "THIS IS X ALL HEAD", X_all.head()
    y_all= train['outcome']
    print "THIS IS Y ALL HEAD",  y_all.head()
    test_x=test[col_list_test]    
    
    #Drop NAs
    X_all=X_all.dropna(axis=1)
    test_x=test_x.dropna(axis=1)
    
    for col in X_all.columns.values:
        if col not in test_x.columns.values:
            X_all=X_all.drop(col, axis=1)
    for col in test_x.columns.values:
        if col not in test_x.columns.values:
            test_x=test_x.drop(col, axis=1)
    
    #fit a gradient boosting classifier
    from sklearn.ensemble import GradientBoostingClassifier
    
    clf=Pipeline([("select",VarianceThreshold()), 
                  ("linear_regression",LinearRegression())])
    
    #clf = GradientBoostingClassifier()#before had n_estimators=100,max_features=20
    clf.fit(X_all,y_all)
    
    #predict the new outcome variale
    test['outcome']=clf.predict(test_x)
    
    
    return test['outcome']
  
    
    
    

def missing_value_model(train,test):
   
    
    # Subset train dataframe randomly    
    train=train.sample(frac=0.05, replace=False)  
    
    #Put aside outcome variable to ignore preprocessing and append later    
    outcome=train["outcome"]
    print "THIS IS OUTCOME", outcome.head()
    
    train=train.drop(['outcome'],axis=1)
    
    #Reduce test set to those with missing values
    test_missing_set=test[test['outcome'].isnull()]
    
   
    #drop the extra columns created prior
    test_missing_set=test_missing_set.drop(['outcome'],axis=1)
    
    
    #drop the additional artificially created columns
    train=train.drop(['date_act_fillfw', 'date_act_fillbw', 'group_fillfw','group_fillbw'],axis=1)
    
    #drop other columns already used in the leak solution above as they may potentially skew result here
    train=train.drop(['people_id'],axis=1) #,'group_1','date','date_act'
    test_missing_set=test_missing_set.drop(['people_id'],axis=1) #,'group_1','date','date_act'
    print "This is the head of test_missing_Set after PREP", test_missing_set.head()        
    
    print "THIS IS TRAIN HEAD AND DESCRIBE BEFORE PROCESSING", train.head()
    display(train.describe())
    #encode labels
    #THIS PART NOT WORKING, STIL NEED TO GET To PCA after this
    
    
    train_copy=MultiColumnLabelEncoder().fit_transform(train)    
    
    print "THIS IS TRAIN HEAD AND DESCRIBE AFTER LABEL ENCODING", train_copy.head()
    display(train_copy.describe())
   
    #Getting closer here but still failing to preprocess data correctly, seems to produce many NaNs and 0 counts, not sure why
    train_copy=pd.DataFrame(Imputer(missing_values="NaN",strategy="median",axis=0).fit_transform(train_copy))
    print "THIS IS TRAIN HEAD AND DESCRIBE AFTER IMPUTINNG", train_copy.head()
    display(train_copy.describe())
    train_copy=pd.DataFrame(Normalizer().fit_transform(train_copy))
    print "THIS IS TRAIN HEAD AND DESCRIBE AFTER NORMALIZING", train_copy.head()
    display(train_copy.describe())
    
    train_copy.columns=list(train.columns.values)
    #for colindex,colname in enumerate(list(train.columns.values)):
     #   train[colname]=train_copy[colindex]
     
    print "THIS IS TRAIN HEAD AND DESCRIBE AFTER RENAMING", train_copy.head()
    #update train to the new version    
    train=pd.DataFrame(train_copy)
    print "THIS IS TRAIN HEAD AND DESCRIBE AFTER RENAMING2", train.head()
    
   
    test_missing_set=MultiColumnLabelEncoder().fit_transform(test_missing_set) 
    #inactivating the reassignment of test_missing_set_copy to test_missing_Set until we decide to use PCA
    '''
    test_missing_set_copy=pd.DataFrame(Imputer(missing_values= "NaN",strategy="median",axis=0).fit_transform(test_missing_set))
    test_missing_set_copy=pd.DataFrame(Normalizer().fit_transform(test_missing_set_copy))
    for colindex,colname in enumerate(list(test_missing_set.columns.values)):
        test_missing_set[colname]=test_missing_set_copy[colindex]
    test_missing_set_copy.columns=list(test_missing_set.columns.values)
    test_missing_set=pd.DataFrame(test_missing_set_copy)
    '''
    
    #Add train outcome back in
    train['outcome']=outcome
    #split into training and testing sets
    print "THESE ARE THE NEW COLUMNS", train.columns.values
    feature_cols=list(train.columns[:].drop(['outcome']))
    target_col=outcome
  
    X_all= train[feature_cols]
    print "THIS IS THE HEAD OF X_ALL", X_all.head()
    
    y_all= target_col
    y_all.columns=['outcome']
    
    print "THIS IS THE HEAD OF Y_ALL \n", y_all.head()
    #USE PCA now to identify the most important parts of train data
    #Inativate PCA for now
    '''
    pca=PCA(n_components=2)
    pca.fit(X_all)
    pca_results=rs.pca_results(X_all,pca)
    print pca_results['Explained Variance'].cumsum()
    #first two PCA dimensions cover 90%+ of variance so we'll look at those
    '''
   
    #Train Test Split
  
    #fit a gradient boosting classifier
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()#before had n_estimators=100,max_features=20
    clf.fit(X_all,y_all)
    
    #predict the new outcome variale
    test_missing_set['outcome']=clf.predict(test_missing_set)
    
    #Left join the new test_missing_set to test
    test_missing_set=test_missing_set.reset_index()
    print "THIS IS THE HEAD TEST MISSING SET", test_missing_set.head()
    test=test.reset_index()
    
    
    #subset test_missing_set to only the values of interest (activity_id and outcome)
    test_missing_set=test_missing_set[['activity_id','outcome']]
    
    #merge those to the original test dataset
    test = pd.merge(test, test_missing_set, how="left", on='activity_id',suffixes=('','_new'))
    #fill NAs with blanks and then combine the two columns into a final outcome column
    test['outcome']=test[['outcome']].fillna(float(0.0))
    test['outcome_new']=test[['outcome_new']].fillna(float(0.0))
    #print 'THESE ARE THE DATATYPES', test['outcome'].dtype, test['outcome_new'].dtype
    test['outcome']=pd.to_numeric(test['outcome'])+pd.to_numeric(test['outcome_new'])
    #print "This is the test file after filling NAs", test.iloc[220:230,52:]    
        
    return test

def main():

    # Write a benchmark file to the submissions folder
    benchmark_model().to_csv("benchmark_submission2.csv", index=False)
    print " I POOPED A MODEL"
if __name__ == "__main__":
    main()