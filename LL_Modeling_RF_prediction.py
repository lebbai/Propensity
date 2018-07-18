
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
##init_notebook_mode(connected=True) 
from sklearn import preprocessing

import itertools

import datetime as dt
import warnings
#warnings.filterwarnings(action='once')
warnings.simplefilter('ignore')
from ggplot import *
# random forest classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; 
##sns.set(color_codes=True)
import matplotlib.pyplot as plt
##plt.show()
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
from plottings import *
# ### Handle Functions

### stackoverflow: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
### input : dataframe df
### output : one hot encoded dataframe, with the dictionary for reverse mapping
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

##read challenge data
df_all_final_challenge = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/challenge_mod1.csv')
del df_all_final_challenge['TENURE_D']

##ens2 gives the better prediction 
 

##X_train, X_test, y_train, y_test = train_test_split( X_f, y, test_size=0.3, random_state=RANDOM_STATE)

##models and importan feature to picle
import pickle
# save the model to disk
model_file = '/Users/razzak_lebbai/Desktop/propensity/data/models/Random_forest_v3.pk'
rf_model = pickle.load(open(model_file, 'rb'))

feature_list = '/Users/razzak_lebbai/Desktop/propensity/data/models/feature_list.pk'
loaded_feature = pickle.load(open(feature_list, 'rb'))

df_final_challenge=df_all_final_challenge.loc[:, loaded_feature]
y_pred_challenge = rf_model.predict(df_final_challenge.loc[:, loaded_feature])
prob_challenge = rf_model.predict_proba(df_final_challenge.loc[:, loaded_feature])


final_results = pd.DataFrame() 
##final_results['True_value']=y_test
final_results['Pred_value']=y_pred_challenge
final_results['Probability']=prob_challenge[:,1]



### rename the levels in age
def prop_group(x):
    if x >0.9 and x <=1.0:
        return 1
    
    elif x>0.8 and x <=0.9:
        return 2 
    
    elif x>0.7 and x <=0.8:
        return 3 
    
    elif x>0.6 and x <=0.7:
        return 4
    
    elif x>0.6 and x <=0.5:
        return 5
    
    elif x>0.5 and x <=0.4:
        return  6 
    
    elif x>0.4 and x <=0.3:
        return 7 
    
    elif x>0.3 and x <=0.2:
        return 8 
    
    elif x>0.2 and x <=0.1:
        return 9

    else:
        return 10


##final_results.groupby(pd.cut(final_results["Probability"], np.arange(0, 1, 0.1)))


final_results['segmentation'] = final_results.Probability.apply(lambda x: prop_group(x))

final_results_model1 = pd.concat([df_all_final_challenge.loc[:,['ACCOUNT_GUID','LL_FL']], final_results], axis=1)
final_results_model1.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/output/rf_final_result.csv', index=False)
