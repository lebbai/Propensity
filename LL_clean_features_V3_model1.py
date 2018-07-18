
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
##init_notebook_mode(connected=True) 



import pandas as pd
import numpy as np
import datetime as dt
import warnings
#warnings.filterwarnings(action='once')
warnings.simplefilter('ignore')
from ggplot import *
# random forest classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; 
sns.set(color_codes=True)
import matplotlib.pyplot as plt
plt.show()
get_ipython().magic('matplotlib inline')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout, Bar
import plotly.offline as offline
import plotly.graph_objs as go
from tabulate import tabulate



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
RANDOM_STATE = 123
from sklearn.metrics import roc_curve, auc
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
# ### Handle Functions


#------------------------------------------------------------------------------
# accept a dataframe, remove outliers, returns the lower and upper limits of IQR
#------------------------------------------------------------------------------
def remove_outlier(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return {'low':fence_low, 'high':fence_high}


def pltRelClk(df,feature,pred,varnm,varnmtitle):
    df1=pd.DataFrame(df[pred].groupby(df[feature]).mean().reset_index(name='mean'))
    fig, ax = plt.subplots(figsize=(15,6))
    b = sns.regplot(x=df1[feature], y=df1['mean'], color="g")
    b.axes.set_title(varnm+" "+ varnmtitle,fontsize=30)
    b.set_xlabel(varnm,fontsize=20)
    b.set_ylabel("Click Rate",fontsize=20)
    b.tick_params(labelsize=15)
    return(b)
    


### drops a level from a categorical column in a dataframe
### returns the dataframe 

def remLev(df,colNm,levNm):
    df[colNm] = df[colNm].astype('category')
    dfn = df.loc[df[colNm] != levNm,]
    levo = df[colNm].cat.categories.tolist()
    levn = [x for x in levo if x != levNm]
    dfn[colNm]= pd.Categorical(dfn[colNm], categories = levn)
    return(dfn)    


### stackoverflow: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
### input : dataframe df
### output : one hot encoded dataframe, with the dictionary for reverse mapping
from sklearn.preprocessing import LabelEncoder

def oneHotEncode2(df, le_dict = {}):
    if not le_dict:
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        train = True;
    else:
        columnsToEncode = le_dict.keys()   
        train = False;

    for feature in columnsToEncode:
        if train:
            le_dict[feature] = LabelEncoder()
        try:
            if train:
                df[feature] = le_dict[feature].fit_transform(df[feature])
            else:
                df[feature] = le_dict[feature].transform(df[feature])

            df = pd.concat([df, 
                              pd.get_dummies(df[feature]).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
            df = df.drop(feature, axis=1)
        except:
            print('Error encoding '+feature)
            #df[feature]  = df[feature].convert_objects(convert_numeric='force')
            df[feature]  = df[feature].apply(pd.to_numeric, errors='coerce')
    return (df, le_dict)


import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
 
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


##read cleaned support data
support_df=    pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/ll_support_clean.csv')
del support_df['LL_FL']
# ### Read the dataset

## load ''ll_mod2.pkl'
##dfll = pd.read_pickle('ll_mod_0501.pkl')
dfll = pd.read_pickle('/Users/razzak_lebbai/Desktop/propensity/data/ll_mod_0604.pkl')


dfll.shape



### check for null values
dfll.isnull().sum()


# ## Strip ID fields and get a sample for modeling


ids = ['ACCOUNT_GUID',
      'PSN']
keep =  [col for col in dfll.columns if col not in ids]


ids_df= dfll[ids]
df_id_rem = dfll[keep]


# make a copy
df_all = df_id_rem
df_all.is_mobile = df_all.is_mobile.astype(str)

# =============================================================================
# do some outlier novality detection on non successful customers who may in successful customer list


df_all_final, le_dict_final = oneHotEncode2(df_all)
df_all_final_1= pd.concat([df_all_final, dfll.loc[:,'ACCOUNT_GUID']], axis=1)

##add both support data and final data

df_all_final_2 = df_all_final_1.merge(support_df,on='ACCOUNT_GUID', how='left')

##get guid id for later use
df_all_final_guid = df_all_final_2.loc[:,['ACCOUNT_GUID']]
#
#
#del df_all_final_2['ACCOUNT_GUID']

##take 30% of the data for traing AND 70% challenge
df_30_training = df_all_final_2.sample(frac=0.3,random_state=RANDOM_STATE)
df_70_challenge = df_all_final_2[~df_all_final_2.index.isin(df_30_training.index)]

##save training and challenge data to csv
##this will be use all other moidels

df_30_training.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/total_training_sample', index=False)
df_70_challenge.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/total_challenge_sample', index=False)

del df_30_training['ACCOUNT_GUID']
del df_70_challenge['ACCOUNT_GUID']

##droped catogrical varaibles



df_all_final_lable1 = df_30_training[df_30_training.LL_FL==1]
df_all_final_lable1 = df_all_final_lable1.drop('LL_FL', axis=1)


df_all_final_lable0 = df_30_training[df_30_training.LL_FL==0]

df_all_final_lable0 = df_all_final_lable0.drop('LL_FL', axis=1)
##shuffle the lable 0
df_all_lable0_final = df_all_final_lable0.sample(frac=1.0,random_state=1)

##create a Novelity outlier detection
#create model on successful data

from sklearn.model_selection import train_test_split
from sklearn import svm
##X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
mask= np.random.rand(len(df_all_final_lable1))<0.7

X_train1 =df_all_final_lable1[mask]
##del X_train1['ACCOUNT_GUID']
X_test1 =df_all_final_lable1[~mask]
##del X_test1['ACCOUNT_GUID']

"""
# fit the model
clf = svm.OneClassSVM(nu=0.008, kernel="rbf", gamma=0.00001)
clf.fit(X_train1)
y_pred_train = clf.predict(X_train1)
y_pred_test = clf.predict(X_test1)
##y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size


error_train=[]
error_test=[]
for gamma in np.arange(0.0005,0.02,0.001):
    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=gamma)
    clf.fit(X_train1)
    y_pred_train = clf.predict(X_train1)
    y_pred_test = clf.predict(X_test1)
    ##y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    
    error_train.append(n_error_train/y_pred_train.shape[0])
    
    error_test.append(n_error_test/y_pred_test.shape[0])
   
    
   
    
plt.plot()


rng = np.random.RandomState(42)


from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# define two outlier detection tools to be compared
classifiers = {
    "One-Class SVM": svm.OneClassSVM(nu=0.01,
                                     kernel="rbf", gamma=0.01),
    "Robust covariance": EllipticEnvelope(contamination=0.01),
    "Isolation Forest": IsolationForest(max_samples='auto',
                                        contamination=0.01,
                                        random_state=rng),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=35,
        contamination=0.01)}
    
 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=100)
  
error_n_train_dict={}
error_n_test_dict={}  
for i, (clf_name, clf) in enumerate(classifiers.items()):
#        # fit the data and tag outliers
       if clf_name == "Local Outlier Factor":
           pass
#            clf.fit(X_train)
#            y_pred_train = clf.predict(X_train)
#            y_pred_test = clf.predict(X_test)
#            n_error_train = y_pred_train[y_pred_train == -1].size
#            n_error_test = y_pred_test[y_pred_test == -1].size
#
#            ##y_pred = clf.fit_predict(X)
#            scores_pred = clf.negative_outlier_factor_
#            n_error_train = y_pred_train[y_pred_train == -1].size
#            n_error_test = y_pred_test[y_pred_test == -1].size
#            error_n_train_dict[clf_name]=n_error_train
#            error_n_test_dict[clf_name]=n_error_test
#
#
            
       else:
            clf.fit(X_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            n_error_train = y_pred_train[y_pred_train == -1].size
            n_error_test = y_pred_test[y_pred_test == -1].size


            scores_pred = clf.decision_function(X_train)
            ##n_error_train = y_pred_train[y_pred_train == -1].size
            ##n_error_test = y_pred_test[y_pred_test == -1].size

        
        ##threshold = stats.scoreatpercentile(scores_pred,
         ##                                   100 * 0.01)
        
            ##n_error_train = y_pred_train[y_pred_train == -1].size
            ##n_error_test = y_pred_test[y_pred_test == -1].size
            error_n_train_dict[clf_name]=n_error_train
            error_n_test_dict[clf_name]=n_error_test

plt.plot(np.arange(0.001,0.04,0.001),error_train)
plt.plot(np.arange(0.001,0.04,0.001),n_error_test)
##plt.plot(np.arange(0.005,0.1,0.005),error_test)
       ## n_errors = (y_pred != ground_truth).sum()
# =============================================================================

"""

clf_all = svm.OneClassSVM(nu=0.008, kernel="rbf", gamma=0.00001)
clf_all.fit(df_all_final_lable1)

pedict_lable0= clf_all.predict(df_all_lable0_final)

df_all_lable0_final['outlier_detection']=pedict_lable0

##now take outlier as 0 

##get the index of zeros aka ouliers from the prediction
get_index= df_all_lable0_final[df_all_lable0_final.outlier_detection==-1].index

##get zero from oroginal data
df_all_final_sampled_lable0 = df_all_final_lable0[df_all_final_lable0.index.isin(get_index)==True]


challenge_index = df_70_challenge.index
df_all_final_challenge_1 = pd.concat([df_70_challenge, df_all_final_guid[df_all_final_guid.index.isin(challenge_index)==True]], axis=1)



####now mix 0 and 1 to create training data set
df_all_final_sampled_lable0['LL_FL']=0
df_all_final_lable1['LL_FL']=1
df_train = pd.concat([df_all_final_sampled_lable0,df_all_final_lable1])
df_train =  df_train.sample(frac=1) 

##get the original features for df_train
df_train_original =df_all[df_all.index.isin(df_train.index)==True]
###now apply OneClassSVM on entire lable 1 data set

##save all data in different files 
df_train.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/train_mod1.csv', index=False)
df_train_original.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/train_ori_feat_mod1.csv', index=False)
df_all_final_challenge_1.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/challenge_mod1.csv', index=False)
