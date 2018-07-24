#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 12:30:29 2018

@author: razzak_lebbai

This fill has two classes one is for the usefull functions
Other one for the multiple models 
Model optimization test shouhd be done seperrately
"""
    

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
import itertools
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import pickle
class Plots():
        
    def plot_confusion_matrix2(self,cm, labels, title='Confusion matrix', cmap=plt.cm.Blues, path='/Users/razzak_lebbai/junk/test.png'):
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
    
    
    def plot_confusion_matrix1(self, cm, classes,
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
    


    
class Functions:
    def __init_(self):
        print('get funtions')
    
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
            
       
        
  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
RANDOM_STATE = 123
class SymPropModels:
    def __init__(self, trainData='train.csv', challengeData='test.csv'):
        self.trainData = trainData
        self.challengeData = challengeData
        print(trainData,challengeData )
        
    
    def ReadTrain(self):
        df=pd.read_csv(self.trainData)
        try:
            del df['TENURE_D']
        except:
            pass
        return df
   
    def createXy(self, df, label='LL_FL'):
        X= df.loc[:, df.columns!=label]
        y= df.loc[:, df.columns==label]
        
        return (X,y)
        
        
        
    ###do feature important studies
    def GetImportanetFeatures(self,df, n_estimators=100, rank_value=0.0001,\
                              path='/Users/razzak_lebbai/Desktop/propensity/src/feature/model/plots/', \
                              filename='feature_importance.png',\
                              feature_list_path = '/Users/razzak_lebbai/Desktop/propensity/data/models/feature_list.pk'): ##X_ is features with no correlations 
        ##RANDOM_STATE = 123     
        data=self.createXy(df)
        X=data[0]
        y=data[1]
        features = X.columns
        
        #####################Start Feature extraction #############
    
        # 
        forest = RandomForestClassifier(n_estimators=n_estimators,
                                      random_state=43)
        
                                      
        forest.fit(X, y)
        
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        all_features=[]
        for f in range(X.shape[1]):
            print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
            all_features.append([f+1, features[indices[f]], importances[indices[f]]])
        
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        ##plt.xticks(range(X.shape[1]), indices)
        plt.xticks(range(X.shape[1]), range(1, (X.shape[1] +1)))
        plt.xlim([-1, X.shape[1]])
        plt.savefig(path+filename)
        plt.show()
        
        
        ##reject useless features
        filter_fea= np.where(importances>rank_value)
        imp_features = features[filter_fea]
        pickle.dump(imp_features, open(feature_list_path, 'wb'))
        
        return imp_features, all_features
 
    def MLModels(self):
        
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
        models={'lr':clf0, 'sv':clf1, 'rf': clf2, 'gb': clf3, 'gnb': clf4,'knn':clf5, 'ens1':eclf1, 'ens2':eclf2, 'ens3':eclf3}
        
        return models



    
    ##this function run and create models and performances
    def test_Models(self,X, y, model_name='rf'):
        
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=RANDOM_STATE)
        all_models = self.MLModels()
        model = all_models[model_name]

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
        Plots().plot_confusion_matrix1(cm, classes=labels, normalize=False,
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
        Plots().plot_confusion_matrix1(cm_normalized, classes=labels, normalize=True,
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


    def readChallngeData(self, file='/Users/razzak_lebbai/Desktop/propensity/data/total_challenge_sample'):
        
        return pd.read_csv(file)
    
    def GetModel(self,X,y, ml_mode='rf',\
                  ouput_file='/Users/razzak_lebbai/Desktop/propensity/data/models/Random_forest_v3.pk'):
        all_models = self.MLModels()
        model = all_models[ml_mode]
        model.fit(X, y)
        ###models and importan feature to picle

        ## save the model to disk
        pickle.dump(model, open(ouput_file, 'wb'))
        
        return model

###

#      
#        
#    def PredictChallene(self,X,y, df_challenge, ml_mode='rf' ):
#        all_models = self.MLModels()
#        model = all_models[ml_mode]
#        model.fit(X, y)
#
#        X_challenge=df_challenge.loc[:, X.columns]
#        y_pred_challenge = model.predict(X_challenge)
#        prob_challenge = model.predict_proba(X_challenge)
#
#
#        final_results = pd.DataFrame() 
#        ##final_results['True_value']=y_test
#        final_results['Pred_value']=y_pred_challenge
#        final_results['Probability']=prob_challenge[:,1]
#        
#        return final_results
    
    def PredictChallene(self, model, df_challenge, imp_feat):
        X_challenge=df_challenge.loc[:, imp_feat]
        y_pred_challenge = model.predict(X_challenge)
        prob_challenge = model.predict_proba(X_challenge)


        final_results = pd.DataFrame() 
        ##final_results['True_value']=y_test
        final_results['Pred_value']=y_pred_challenge
        final_results['Probability']=prob_challenge[:,1]
        
        return final_results
    

    def GetOneClassModel(self, X, nu=0.005,kernel="rbf", gamma=0.00001 ):
        clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        clf.fit(X)
        
        
        return clf
    
        
        
    def OneClassSVMPrdict(self, model, df_challenge, imp_feat):
        X_challenge=df_challenge.loc[:, imp_feat]
        y_pred = model.predict(X_challnge)
        scores_pred_train = model.decision_function(X_challnge)
        
        
        final_results = pd.DataFrame() 
        ##final_results['True_value']=y_test
        final_results['Pred_value']=y_pred
        final_results['decision_function']=scores_pred_train
        
        return final_results
    
    

   

#
#
## define two outlier detection tools to be compared
#classifiers = {
#    "One-Class SVM": svm.OneClassSVM(nu=0.01,
#                                     kernel="rbf", gamma=0.01),
#    "Robust covariance": EllipticEnvelope(contamination=0.01),
#    "Isolation Forest": IsolationForest(max_samples='auto',
#                                        contamination=0.01,
#                                        random_state=rng)
#     
#
#        
#        
#        





      
        
train_path='/Users/razzak_lebbai/Desktop/propensity/data/train_mod1.csv'      
mod1=SymPropModels(trainData=train_path)

df = mod1.ReadTrain()
##mod1.GetImportanetFeatures(df)

X,y=  mod1.createXy(df)
##get the inportantr featurews
imp_feat, all_feature = mod1.GetImportanetFeatures(df, rank_value=0.0001)

model_rf= mod1.GetModel(X.loc[:,imp_feat],y)


##mod1.test_Models(X.loc[:,imp_feat], y, model_name='rf')

##read the data

df_challenge = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/total_challenge_sample')

final_result = mod1.PredictChallene(model_rf,df_challenge,imp_feat )