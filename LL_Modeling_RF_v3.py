
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


def catogarical(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        for f in columnsToEncode:
            df[f] = pd.Categorical(df[f])
            df[f+'_code'] = df[f].cat.codes
        
        return df, columnsToEncode
        
            

def LabelEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        encode_map={}
        for f in columnsToEncode:
            le = preprocessing.LabelEncoder()
            le.fit(df[f])
            
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            encode_map[f]=le_name_mapping
            
            df[f+'_encode'] = le.transform(df[f])
        
        return df, columnsToEncode,encode_map
        
            
     
        
 
##read train data
df_train = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/train_mod1.csv')
del df_train['TENURE_D']

##read origina d features for train
df_train_original =pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/train_ori_feat_mod1.csv')
del df_train_original['TENURE_D']

df_train_original_cat, cols, endcode_mapping = LabelEncode(df_train_original)
df_train_original_cat = df_train_original_cat.drop(cols, axis=1)
##correlation study

##read challenge data
df_all_final_challenge = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/challenge_mod1.csv')
del df_all_final_challenge['TENURE_D']

X= df_train.loc[:, df_train.columns!='LL_FL']
y=df_train.loc[:, df_train.columns=='LL_FL']

###do feature important studies
def GetImportanetFeatures(X_, y, rank_value=0.0001, path='/Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/feature_importance.png'): ##X_ is features with no correlations 
    ##RANDOM_STATE = 123     
    features = X_.columns
    
    #####################Start Feature extraction #############

    # 
    forest = RandomForestClassifier(n_estimators=122,
                                  random_state=43)
    
                                  
    forest.fit(X_, y)
    
    
    
    
    
    
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    all_features=[]
    for f in range(X_.shape[1]):
        print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
        all_features.append([f+1, features[indices[f]], importances[indices[f]]])
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    ##plt.xticks(range(X.shape[1]), indices)
    plt.xticks(range(X_.shape[1]), range(1, (X_.shape[1] +1)))
    plt.xlim([-1, X_.shape[1]])
    plt.savefig(path)
    plt.show()
    
    
    ##reject useless features
    filter_fea= np.where(importances>rank_value)
    imp_features = features[filter_fea]
    
    return imp_features, all_features



def getNumberOfEstimaters(X, y, n_esti=100):
    
    ##forest_test = RandomForestClassifier(n_estimators=20, class_weight= 'balanced', random_state=43)
    scores = []
    for val in range(2,n_esti, 2):
        forest_test = RandomForestClassifier(n_estimators=val,random_state=43)
        validated = cross_val_score(forest_test,X, y, cv=10)
        scores.append([validated.mean(), val ])
    #    
    
    numpy_score = np.array(scores)
    
    optimized_estimators = numpy_score[np.argmax(numpy_score[:,0]), 1]
    
    return optimized_estimators, numpy_score
    #optimized_cluster=68
       


def plot_confusion_matrix2(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues, path='/Users/razzak_lebbai/junk/test.png'):
        fig = plt.figure(figsize=(8,8))
    #    fig = plt.figure()
        ax  = fig.add_subplot(111)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        width, height = cm.shape
        ax=plt.gca()
    #    PCM=ax.get_children()[2]
        for x in range(width):
            for y in range(height):
                ax.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
        
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(path)


def plot_confusion_matrix1(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def GetTrainingPerformance(X,y, nest=80 ):
     
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)
    
    scaler = preprocessing.StandardScaler().fit(X_train)
#
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#
#
#
    rf_train = RandomForestClassifier(n_estimators=nest, class_weight= 'balanced',\
                                      random_state=43,  max_features=None)
    rf_train.fit(X_train, y_train)
    y_pred = rf_train.predict(X_test)
    prob = rf_train.predict_proba(X_test)
    
    ##y_pred1 = (prob[:, 1]>0.70).astype(int)
    
    y_score = rf_train.score(X_test, y_test)
    
    
    
    ##labels = ['Won business', 'Lost Business']
    labels = rf_train.classes_.tolist()

    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    ##np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, cmap=plt.cm.OrRd, path='Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/Confusion_matrix.png', labels=labels)
    plt.show()
    
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix', cmap=plt.cm.OrRd, path='Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/Normalized_Confusion_matrix.png', labels=labels)
    
    plt.show()
    
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr_rf, tpr_rf)
    
    
    #
    ## Compute ROC curve and ROC area for each class
    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()
    #for i in range(2):
    #    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #    roc_auc[i] = auc(fpr[i], tpr[i])
    #
    ## Compute micro-average ROC curve and ROC area
    #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    ##plt.plot(fpr_rf, fpr_rf, label='ROC')
    ##plt.plot(fpr_rf, fpr_rf, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr_rf, tpr_rf, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig('Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/ROC_curve.png')
    plt.show()
    
    accuray = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1) 
    recall = recall_score(y_test, y_pred,  pos_label=1) 
    
    #precision_label = precision_score(y_test, y_pred, average= None) 
    #recall_label = recall_score(y_test, y_pred, average= None) 
    F1_score = f1_score(y_test, y_pred, average='binary')
    print("Accuracy %0.3f" %accuray)
    print("precision %0.3f" %precision)
    print("recall %0.3f" %recall)
    print("f1_score %0.3f" %F1_score)

##GetTrainingPerformance(X,y)
    
#    
##create several algoirthms 
##these vrainbles were tested 

clf0 = LogisticRegression(random_state=RANDOM_STATE)
clf1 = SVC(C=1.0, kernel='rbf', probability=True, gamma=0.015)
clf2 = RandomForestClassifier(n_estimators=170, class_weight= 'balanced',\
                                      random_state=RANDOM_STATE,  max_features=None, n_jobs=6)
clf3 =GradientBoostingClassifier(n_estimators= 1200, max_leaf_nodes= 4, max_depth= None, random_state= RANDOM_STATE, min_samples_split= 5)
clf4 = GaussianNB()
clf5 = KNeighborsClassifier(n_neighbors=17, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=3)

###get multiple ensample a

eclf1 = VotingClassifier(estimators=[ ('sv', clf1), ('rf', clf2), ('gb', clf3), ('gnb', clf4)],
                        voting='soft', weights=[1,1,1,1])
eclf2 = VotingClassifier(estimators=[ ('rf', clf2), ('gb', clf3), ('gnb', clf4)],
                        voting='soft', weights=[1,1,1])

eclf3 = VotingClassifier(estimators=[ ('rf', clf2), ('gb', clf3)],
                        voting='soft', weights=[1,1])




X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=RANDOM_STATE)

 
"""
###the following codes will do tests to optimise the algorithms
##d=for random forest
test_error_est =[]
train_error_est =[]
n_esti=200
for est in range(130,n_esti, 10):
    
    rf_train = RandomForestClassifier(n_estimators=est, class_weight= 'balanced',\
                                      random_state=RANDOM_STATE,  max_features=None)
    rf_train.fit(X_train, y_train)
    y_pred_test = rf_train.predict(X_test)
    y_pred_train = rf_train.predict(X_train)
    msqe_test= mean_squared_error(y_test, y_pred_test)
    msqe_train= mean_squared_error(y_train, y_pred_train)
    test_error_est.append([est, msqe_test])
    train_error_est.append([est, msqe_train])
   
plt.plot(np.array(test_error_est)[:,0], np.array(test_error_est)[:,1], label='test')
plt.plot(np.array(train_error_est)[:,0], np.array(train_error_est)[:,1], label='train')
plt.legend(loc="upper right")
test_error_est1 = test_error_est
train_error_est1 = train_error_est

####################


##for SVM

test_error_est =[]
train_error_est =[]
g_max=0.1
for g in np.arange(0.001,g_max, 0.002):
    
    rf_train = SVC(C=1.0, kernel='rbf', cache_size=800, probability=False, gamma=g)
    rf_train.fit(X_train, y_train)
    y_pred_test = rf_train.predict(X_test)
    y_pred_train = rf_train.predict(X_train)
    msqe_test= mean_squared_error(y_test, y_pred_test)
    msqe_train= mean_squared_error(y_train, y_pred_train)
    test_error_est.append([g, msqe_test])
    train_error_est.append([g, msqe_train])
    
    
plt.plot(np.array(test_error_est)[:,0], np.array(test_error_est)[:,1], label='test')
plt.plot(np.array(train_error_est)[:,0], np.array(train_error_est)[:,1], label='train')
plt.legend(loc="upper right")


##for GradientBooost
test_error_est =[]
train_error_est =[]

n_esti=2000
for est in range(1000,n_esti, 100):
    
    rf_train = GradientBoostingClassifier(n_estimators= est, max_leaf_nodes= 4, max_depth= None, random_state= RANDOM_STATE, min_samples_split= 5)
    rf_train.fit(X_train, y_train)
    y_pred_test = rf_train.predict(X_test)
    y_pred_train = rf_train.predict(X_train)
    msqe_test= mean_squared_error(y_test, y_pred_test)
    msqe_train= mean_squared_error(y_train, y_pred_train)
    test_error_est.append([est, msqe_test])
    train_error_est.append([est, msqe_train])
    
plt.plot(np.array(test_error_est)[:,0], np.array(test_error_est)[:,1], 'ob',label='test')
plt.plot(np.array(train_error_est)[:,0], np.array(train_error_est)[:,1],'or', label='train')
plt.legend(loc="upper right")

"""



##this function run and create models and performances
def test_Models(X, y, model_name='rf'):
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=RANDOM_STATE)
       
    model = models[model_name]
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    msqe_test= mean_squared_error(y_test, y_pred_test)
    msqe_train= mean_squared_error(y_train, y_pred_train)
        ##labels = ['Won business', 'Lost Business']
    labels = model.classes_.tolist()


    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    ##np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    ##plot_confusion_matrix2(cm, cmap=plt.cm.OrRd, title='Confusion matrix for '+model_name, path='/Users/razzak_lebbai/Desktop/omniture/src/analysis/v1/plots/'+model_name+'_Confusion_matrix.png', labels=labels)
    plot_confusion_matrix1(cm, classes=labels, normalize=False,
                    title='Normalized confusion matrix')
    plt.savefig('/Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/'+model_name+'_con_matrxi.png')

    ##plot_confusion_matrix(cm)
    plt.show()
    
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    ##plot_confusion_matrix2(cm_normalized, title='Normalized confusion matrix for '+model_name, cmap=plt.cm.OrRd, path='/Users/razzak_lebbai/Desktop/omniture/src/analysis/v1/plots/'+model_name+'_Normalized_Confusion_matrix.png', labels=labels)
    plot_confusion_matrix1(cm_normalized, classes=labels, normalize=True,
                      title='Normalized confusion matrix')
    plt.savefig('/Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/'+model_name+'_Norm_con_matrxi.png')

    plt.show()
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr_rf, tpr_rf)
      
    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    ##plt.plot(fpr_rf, fpr_rf, label='ROC')
    ##plt.plot(fpr_rf, fpr_rf, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr_rf, tpr_rf, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for '+model_name)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig('/Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/'+model_name+'_ROC_curve.png')
    plt.show()
        
    
    
    ###performance values
    
    accuray = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, pos_label=1) 
    recall = recall_score(y_test, y_pred_test,  pos_label=1) 
    
    #precision_label = precision_score(y_test, y_pred, average= None) 
    #recall_label = recall_score(y_test, y_pred, average= None) 
    F1_score = f1_score(y_test, y_pred_test, average='binary')
    print("Accuracy %0.3f" %accuray)
    print("precision %0.3f" %precision)
    print("recall %0.3f" %recall)
    print("f1_score %0.3f" %F1_score)
    
    return [accuray, precision, recall, F1_score]



models={'lr':clf0, 'sv':clf1, 'rf': clf2, 'gb': clf3, 'gnb': clf4,'knn':clf5, 'ens1':eclf1, 'ens2':eclf2, 'ens3':eclf3}
imp_feat, all_feature = GetImportanetFeatures(X, y, rank_value=0.0001)
X_f=X.loc[:, imp_feat]

##get all features in ranked format
df_features = pd.DataFrame(np.array(all_feature)[:,:2], columns=['ranking', 'features'])
df_features.features = df_features.features.str.replace(r'\_[0-9]$','')
df_features.ranking = df_features.ranking.astype(int)

df_features_rank = df_features.groupby('features').agg({'ranking':'min'}).reset_index()
df_features_rank = df_features_rank.sort_values(['ranking']).reset_index(drop=True)
df_features_rank['ranking_new'] = df_features_rank.index+1
del df_features_rank['ranking']

##################


test_Models(X.loc[:, imp_feat], y, model_name='rf')
"""
##run each algorthms and get the performances 
performance={}
for k in models.keys():
    performance[k]=test_Models(X_f, y, model_name=k)
    
    
    

perf_df =   pd.DataFrame.from_dict(performance)

columns={'lr':'Logistic Reg', 'rf':'Random Forest', 'sv':'SVM', 'gnb':'Gaussian Naive Bayes', 'gb':'Gradient Boosting', 'ens1':'Ensamble1', 'ens2':'Ensamble2','ens3':'Ensamble3' }
perf_df = perf_df.rename(columns=columns)

perf_df['Performance'] = np.array(['accuray', 'precision','recall','f1_score'])
perf_df = np.round(perf_df, decimals=4)
perf_df.to_csv('/Users/razzak_lebbai/Desktop/omniture/src/analysis/v1/plots/performance.csv', index=False)

perf_df =perf_df.set_index('Performance')



##performqnce plots 
for perf in perf_df.index:   
    
    ax = perf_df.loc[perf, :].plot(kind='bar',    # Plot a bar chart
            legend=False,    # Turn the Legend off
            width=0.75,      # Set bar width as 75% of space available
            figsize=(8,5.8),  # Set size of plot in inches
            title=perf,
            color=[plt.cm.tab10(np.arange(perf_df.shape[1]))]
            
            )
    
    
    
    for p in ax.patches:
            ax.annotate(str(p.get_height()*100)+'%', (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.savefig('/Users/razzak_lebbai/Desktop/omniture/src/analysis/v1/plots/'+perf+'.png', bbox_inches="tight")
    plt.show()


###############################

"""
##ens2 gives the better prediction 
 

##X_train, X_test, y_train, y_test = train_test_split( X_f, y, test_size=0.3, random_state=RANDOM_STATE)

  
model = models['rf']
model.fit(X.loc[:, imp_feat], y)

##models and importan feature to picle
import pickle
# save the model to disk
filename = '/Users/razzak_lebbai/Desktop/propensity/data/models/Random_forest_v3.pk'
pickle.dump(model, open(filename, 'wb'))
 

feature_list = '/Users/razzak_lebbai/Desktop/propensity/data/models/feature_list.pk'
pickle.dump(imp_feat, open(feature_list, 'wb'))



###feature comparision studies
for col in  df_train_original.columns:
    if col!='LL_FL':
        if '_encode'  in col: 
            continue
        coutings = df_train_original.groupby(col)['LL_FL'].count()
        BarPlotWithGroupNoAnoNosci(coutings, bbox_to_anchor=(0.95, 0.95), path='/Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/'+coutings.index.name+'_count.png')
        means = df_train_original.groupby(col)['LL_FL'].mean()
        BarPlotWithGroupNoAnoNosci(df_train_original.groupby(col)['LL_FL'].mean(), bbox_to_anchor=(0.95, 0.95), path='/Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/'+means.index.name+'_mean.png')
    else:
        pass
 
