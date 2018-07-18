
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



##df_training = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/total_training_sample')
df_challenge = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/total_challenge_sample')


##get guid id for later use of challenge data
df_challenge_guid = df_challenge.loc[:,['ACCOUNT_GUID']]
#

##models and importan feature to picle
import pickle
# load the model to disk
filename_OneClassSvm = '/Users/razzak_lebbai/Desktop/propensity/data/models/OneSvm_v3_05.pk'

onesvm_model = pickle.load(open(filename_OneClassSvm, 'rb'))
 

feature_list_OneClassSvm = '/Users/razzak_lebbai/Desktop/propensity/data/models/feature_list_onesvm.pk'

onesvm_featues = pickle.load(open(feature_list_OneClassSvm, 'rb'))


X_challenge = df_challenge.loc[:,onesvm_featues]
##df_all_final_lable0_10 = X_challenge.sample(frac=0.1,random_state=1)
pedict_labels= onesvm_model.predict(X_challenge)
scores_pedict_labels = onesvm_model.decision_function(X_challenge)

final_results_cluster = pd.DataFrame() 
##final_results['True_value']=y_test
final_results_cluster['Pred_value']=pedict_labels
final_results_cluster['decision_function']=scores_pedict_labels

final_results_cluseter_model2 = pd.concat([df_challenge.loc[:,['ACCOUNT_GUID','LL_FL']], final_results_cluster], axis=1)
final_results_cluseter_model2.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/output/onesvm_final_result_05.csv', index=False)

