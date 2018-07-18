
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



df_training = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/total_training_sample')
df_challenge = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/total_challenge_sample')


##get guid id for later use of challenge data
df_challenge_guid = df_challenge.loc[:,['ACCOUNT_GUID']]
#

##remove guid for traning purpose
del df_training['ACCOUNT_GUID']
del df_challenge['ACCOUNT_GUID'] 



df_all_final_lable1 = df_training[df_training.LL_FL==1]
df_all_final_lable1 = df_all_final_lable1.drop('LL_FL', axis=1)

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

rng = np.random.RandomState(42)
# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.00001)
clf.fit(X_train1)
y_pred_train = clf.predict(X_train1)
y_pred_test = clf.predict(X_test1)
scores_pred_train = clf.decision_function(X_train1)
scores_pred_test = clf.decision_function(X_test1)

##y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
error_train = n_error_train/X_train1.shape[0]
error_test = n_error_test/X_test1.shape[0]
print("train: {:.3f}, test{:.3f}".format(error_train,error_test))



rc_clf = EllipticEnvelope(contamination=0.05)

rc_clf.fit(X_train1)
y_pred_train_rc = rc_clf.predict(X_train1)
y_pred_test_rc = rc_clf.predict(X_test1)

scores_pred_train_rc = rc_clf.decision_function(X_train1)
scores_pred_test_rc = rc_clf.decision_function(X_test1)

##y_pred_outliers = clf.predict(X_outliers)
n_error_train_rc = y_pred_train_rc[y_pred_train_rc == -1].size
n_error_test_rc = y_pred_test_rc[y_pred_test_rc == -1].size
error_train_rc = n_error_train_rc/X_train1.shape[0]
error_test_rc = n_error_test_rc/X_test1.shape[0]
print("train: {:.3f}, test{:.3f}".format(error_train_rc,error_test_rc))


"""

if_clf = IsolationForest(max_samples='auto', contamination=0.05,
                                        random_state=rng)   
if_clf.fit(X_train1)
y_pred_train_if = if_clf.predict(X_train1)
y_pred_test_if = if_clf.predict(X_test1)


scores_pred_train_if = if_clf.decision_function(X_train1)
scores_pred_test_if = if_clf.decision_function(X_test1)

##y_pred_outliers = clf.predict(X_outliers)
n_error_train_if = y_pred_train_if[y_pred_train_if == -1].size
n_error_test_if = y_pred_test_if[y_pred_test_if == -1].size
error_train_if = n_error_train_if/X_train1.shape[0]
error_test_if = n_error_test_if/X_test1.shape[0]
print("train: {:.3f}, test{:.3f}".format(error_train_if,error_test_if))
"""
"""
lof_clf = LocalOutlierFactor(
        n_neighbors=35,
        contamination=0.05)
    

lof_clf.fit(X_train1)
y_pred_train_lof = lof_clf.predict(X_train1)
y_pred_test_lof = lof_clf.predict(X_test1)

if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            scores_pred = clf.negative_outlier_factor_
##y_pred_outliers = clf.predict(X_outliers)
n_error_train_lof = y_pred_train_lof[y_pred_train_lof == -1].size
n_error_test_lof = y_pred_test_lof[y_pred_test_lof == -1].size
error_train_lof = n_error_train_lof/X_train1.shape[0]
error_test_lof = n_error_test_lof/X_test1.shape[0]
print("train: {:.3f}, test{:.3f}".format(error_train_lof,error_test_lof))
 
"""

"""                                  
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

clf_all = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.00001)
clf_all.fit(df_all_final_lable1)
##models and importan feature to picle
import pickle
# save the model to disk
filename_onesvm = '/Users/razzak_lebbai/Desktop/propensity/data/models/OneSvm_v3_05.pk'
pickle.dump(clf_all, open(filename_onesvm, 'wb'))
 

feature_list_onesvm = '/Users/razzak_lebbai/Desktop/propensity/data/models/feature_list_onesvm.pk'
pickle.dump(df_all_final_lable1.columns, open(feature_list_onesvm, 'wb'))


"""

X_challenge = df_challenge.loc[:,df_all_final_lable1.columns]
##df_all_final_lable0_10 = X_challenge.sample(frac=0.1,random_state=1)
pedict_labels= clf_all.predict(X_challenge)
scores_pedict_labels = clf.decision_function(X_challenge)

final_results_cluster = pd.DataFrame() 
##final_results['True_value']=y_test
final_results_cluster['Pred_value']=pedict_labels
final_results_cluster['decision_function']=scores_pedict_labels
final_results_cluseter_model2 = pd.concat([df_all_final_challenge.loc[:,['ACCOUNT_GUID','LL_FL']], final_results_cluster], axis=1)
final_results_cluseter_model2.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/output/onesvm_final_result.csv', index=False)
"""