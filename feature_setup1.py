#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:18:25 2018

@author: razzak_lebbai
"""

import pandas as pd
import numpy as np

from functions_propencity import *
from matplotlib.patches import Polygon


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
from pandas.tools.plotting import table
##from plottings import *
##from plottings import *
import datetime
from datetime import timedelta
import math
parse_dates=['VISIT_STRT_DTTM_GMT']
##country code list 
## this needs to check landing page with contry codes
cn_code= pd.read_csv('/Users/razzak_lebbai/Desktop/omniture/data/country_code.csv')
cncode_list = cn_code.EVR_27_PRP_2_SITE_CTRY_CD.tolist()

##usewr_agent data after clean in terada
user_agent_df=pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/user_agent_clean.csv')



##LL matched customers
ll_df1 = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/data_v4.csv',  nrows=None)





ll_df1['Landing_web_site']=ll_df1.Landing_page.apply(lambda x: np.nan if pd.isnull(x) else  ParseUrl(x)[1])
 

##catogarize landing pages 
Landing_web_site_df = pd.DataFrame(ll_df1.Landing_web_site.unique(), columns=['page'])
Landing_web_site_keys = Landing_web_site_keys(Landing_web_site_df)




##catogarize lading pages
ll_df1['Landing_web_site_catogary'] = ll_df1.Landing_web_site.apply(lambda x: Landing_web_site_parse1(x, Landing_web_site_keys))
ll_df1.number_session = ll_df1.number_session.fillna(0)

##merger with user_agent data
ll_df2 = ll_df1.merge(user_agent_df, how='left', on=['ACCOUNT_GUID','PSN'])

##catogarize browser family
ll_df2['Browser_Family_catogary'] = ll_df2.Browser_Family.apply(lambda x: np.nan if pd.isnull(x) else convergerBrowser(x))

#catogarise the OS familyu
ll_df2['OS_Family_catogary'] = ll_df2.OS_Family.apply(lambda x: np.nan if pd.isnull(x) else convergerOS(x))

pull_date = '2018-06-04'


ll_df2['LL_FL'].mean()

##remove the expiration is way long or hard to work with
ll_df2 = ll_df2[ll_df2.EXPIRATION_DATE<pd.Timestamp.max.strftime('%Y-%m-%d')]

##change to time data type
ll_df2.ORDER_DATE = lookup(ll_df2.ORDER_DATE)

ll_df2.EXPIRATION_DATE = lookup(ll_df2.EXPIRATION_DATE)
# ## Lifecycle, time to expiration
ll_df2.EXPIRATION_DATE.isnull().sum()

##one year with order to fill out null values for expit=ration date
ll_df2.EXPIRATION_DATE = ll_df2.EXPIRATION_DATE.fillna(ll_df2.ORDER_DATE + datetime.timedelta(days=365))


# ## Lifecycle, time to expiration
ll_df2.EXPIRATION_DATE.isnull().sum()


# data was pulled on- '2018-03-29'
date_end = pd.to_datetime(pull_date, format='%Y-%m-%d', errors='coerce')



# create a feature with study end date
# min(date_end,order_date)
ll_df2['STUDY_DATE_END'] = ll_df2['ORDER_DATE']
ll_df2['STUDY_DATE_END'].loc[ll_df2.LL_FL<1] = date_end


##just making sure no 
ll_df2.EXPIRATION_DATE.isnull().sum() ## 0
ll_df2 = ll_df2[ll_df2.EXPIRATION_DATE.notnull()]
ll_df2.EXPIRATION_DATE.isnull().sum() ## 0

##get days 
ll_df2['DAYS_TO_EXPIRE'] = (ll_df2.EXPIRATION_DATE - ll_df2.STUDY_DATE_END)/np.timedelta64(1, 'D')



# Bucket for exploration


ll_df2['DAYS_TO_EXPIRE_BIN']=pd.cut(ll_df2['DAYS_TO_EXPIRE'],
                             bins = [-200,-60,0,60,120,180,240,300,5000],
                             labels = ['Bad_data','Renewal', '0-2m', '2-4m', '4-6m','6-8m','8-10m','>10m'])



#df1['DAYS_TO_EXPIRE_BIN'].value_counts()
##pd.DataFrame(ll_df2['LL_FL'].groupby(ll_df2['DAYS_TO_EXPIRE_BIN']).mean().reset_index(name='mean'))
ll_df2.groupby('DAYS_TO_EXPIRE_BIN')['LL_FL'].mean().reset_index(name='mean')

test1=ll_df2.groupby('DAYS_TO_EXPIRE_BIN')['LL_FL'].mean().reset_index(name='means')
plt.bar(test1.DAYS_TO_EXPIRE_BIN, test1.means)
######################################################



# ## Active Product Acquisition Channel #############################
#####################################################################

ll_df2['PSN_CHANNEL'].isnull().sum() ### 0  none missing
##ll_df2['LL_FL'].groupby(ll_df2['PSN_CHANNEL']).describe()
ll_df2.groupby('PSN_CHANNEL')['LL_FL'].describe()


##ignore CTO and PPOEM channels

#   
#  - The **LOEM**, **CTO**   subscription channel adoption rates are outliers and should be ignored for modeling.
# 
#
ll_df2['PSN_CHANNEL_N']= np.where(ll_df2.PSN_CHANNEL.isin(['CTO','PPOEM']),'Ignore',ll_df2.PSN_CHANNEL)


 ## PROD_FMLY_GRP
ll_df2.PROD_FMLY_GRP.isnull().sum()   ## 0 NO MISSING ENTRIES


(ll_df2.PROD_FMLY_GRP.value_counts()/ll_df2.shape[0]).plot(color='b')
(ll_df2.PROD_FMLY_GRP[ll_df2.LL_FL==1].value_counts()/ll_df2.PROD_FMLY_GRP[ll_df2.LL_FL==1].shape[0]).plot(color='r')




##cleanup the products family

ll_df2['PROD_FMLY_GRP_CLEANED'] = ll_df2['PROD_FMLY_GRP'].apply(convergerProd)


### BEWARE OF THE LL_STD_ADD & NS-BU-STORAGE PROD FMLY GRP (not use this feature for modeling)
##df1['LL_FL'].groupby(df1['PROD_FMLY_GRP_CLEANED']).describe()
ll_df2.groupby('PROD_FMLY_GRP_CLEANED')['LL_FL'].describe()
######END product family#################################


# ## Current AR Flag###########################################################
# ### Opt-in purchase at a higher rate ########################################
##Impute Missing value ~ 600K & 700K
ll_df2['CURRENT_AR_FLAG'] = ll_df2.CURRENT_AR_FLAG.fillna('Missing')
ll_df2['PURCHASE_AR_FLAG'] = ll_df2.PURCHASE_AR_FLAG.fillna('Missing')
####End purchase Flag



### Indicator if Current Flag is different than Purchase
#df1['CURRENT_PURCHASE_AR_SAME'] = np.where(df1.CURRENT_AR_FLAG == df1.PURCHASE_AR_FLAG,1,0)

## 1: {OPT-OUT,Missing } -> OPT-IN
## 0 : Unchanged
## -1: {OPT-IN} -> {OPT-OUT,Missing }
ll_df2['AR_UPGRADE_PURCHASE'] = np.where(ll_df2.CURRENT_AR_FLAG.isin(['OPT-IN']) & ll_df2.PURCHASE_AR_FLAG.isin(['OPT-OUT','Missing']),1,0)
ll_df2['AR_UPGRADE_PURCHASE'].loc[ll_df2.CURRENT_AR_FLAG.isin(['OPT-OUT','Missing']) & ll_df2.PURCHASE_AR_FLAG.isin(['OPT-IN'])] = -1



##df1['LL_FL'].groupby(df1['AR_UPGRADE_PURCHASE']).describe()
ll_df2.groupby('AR_UPGRADE_PURCHASE')['LL_FL'].describe()

##df1['LL_FL'].groupby(df1['CURRENT_AR_FLAG']).describe()
ll_df2.groupby('CURRENT_AR_FLAG')['LL_FL'].describe()


##df1['CORE_FL'].groupby(df1['CURRENT_AR_FLAG']).describe()
ll_df2.groupby(['CURRENT_AR_FLAG','PSN_CHANNEL_N'])['LL_FL'].describe()

####End AR Purchase##################################################
#####################################################################


# ## ACTIVE PRODUCTS ##################################################
# Number of active products############################################

ll_df2['ACTIVE_PRODUCTS'].isnull().sum()

ll_df2['ACTIVE_PRODUCTS'].describe()


## df1['ACTIVE_PRODUCTS'].isnull().sum() ### 1256529
### the missing active products
miss_imp = 0
ll_df2['ACTIVE_PRODUCTS1'] = ll_df2.ACTIVE_PRODUCTS.fillna(miss_imp)


##find out outlier in active product 
ap_h = ceil(remove_outlier(ll_df2['ACTIVE_PRODUCTS1'])['high'])
#ap_l = 1




ll_df2['ACTIVE_PRODUCTS_PROC'] = ll_df2['ACTIVE_PRODUCTS1'].apply(lambda x: ap_h if x > ap_h else x)
#df1['ACTIVE_PRODUCTS_PROC'] = df1['ACTIVE_PRODUCTS_PROC'].apply(lambda x: round(ap_l) if x < ap_l else x)


##pd.DataFrame(df1['LL_FL'].groupby(df1['ACTIVE_PRODUCTS_PROC']).mean().reset_index(name='mean'))
ll_df2.groupby(['ACTIVE_PRODUCTS_PROC'])['LL_FL'].mean()

ll_df2.groupby(['ACTIVE_PRODUCTS_PROC'])['LL_FL'].mean().plot(kind='bar')
ll_df2.ACTIVE_PRODUCTS_PROC.value_counts(dropna=False)
"""
2.0    4369268
1.0    1493363
0.0    1256529
4.0     526645
3.0     262114
"""

##not sure we need to use thisa or not
ll_df2['ACTIVE_PRODUCTS_PROC1']=pd.cut(ll_df2['ACTIVE_PRODUCTS_PROC'],
                             bins = [-200,0,1,2,5],
                             labels = ['Missing', '1', '2', '>2'])



ll_df2.groupby(['ACTIVE_PRODUCTS_PROC1'])['LL_FL'].mean()

ll_df2.groupby(['ACTIVE_PRODUCTS_PROC1'])['LL_FL'].mean().plot(kind='bar')
###End Active Products
###############################################################################


# ## TENURE (years)#########################################################

ll_df2.START_DATE = lookup(ll_df2.START_DATE)
##ll_df2['START_DATE']=pd.to_datetime(df1.START_DATE, format='%Y-%m-%d', errors='coerce')

ll_df2['START_DATE'].describe()

##get days 
ll_df2['TENURE_D'] = (ll_df2.STUDY_DATE_END - ll_df2.START_DATE)/np.timedelta64(1, 'D')
##df1['TENURE_D'] = [int(i.days) for i in (df1.STUDY_DATE_END - df1.START_DATE)]

ll_df2['TENURE_D'].describe()
ll_df2.TENURE_D.hist()

ll_df2.TENURE_DAYS.isnull().sum() ### 0 missing entries



ll_df2 = ll_df2[ll_df2.TENURE_DAYS.notnull()]


ll_df2['TENURE_YR'] = ll_df2['TENURE_D']/365.0
#ten = remove_outlier(df1['TENURE_YR'])['high']
ten = 10 ### data beyond 10 yrs is not reliable 

ll_df2['TENURE_YR'].describe()


# #### Cap tenure at 10 

ll_df2['TENURE_YR'] = ll_df2['TENURE_YR'].apply(lambda x: 0 if x<0 else round(x))
ll_df2['TENURE_YR'] = ll_df2['TENURE_YR'].apply(lambda x: ten if x>=ten else round(x))


ll_df2.groupby(['TENURE_YR'])['LL_FL'].mean()

ll_df2.groupby(['TENURE_YR'])['LL_FL'].mean().plot(kind='bar')
###Tenure End 
########################################################################


# ## Seat Usage###########################################
ll_df2['SEAT_USAGE'].describe()

out_su = remove_outlier(ll_df2['SEAT_USAGE'])['high']


ll_df2['SEAT_USAGE1'] = ll_df2['SEAT_USAGE'].apply(lambda x: out_su+1 if x > out_su else x)

ll_df2.groupby(['SEAT_USAGE1'])['LL_FL'].mean()

ll_df2.groupby(['SEAT_USAGE1'])['LL_FL'].mean().plot(kind='bar')


### 0 seat usage has very high conversion rate
ll_df2['SEAT_USAGE1'] = ll_df2['SEAT_USAGE1'].apply(lambda x: 1 if x <1 else x)
ll_df2.SEAT_USAGE1.value_counts()

ll_df2.groupby(['SEAT_USAGE1'])['LL_FL'].mean()

ll_df2.groupby(['SEAT_USAGE1'])['LL_FL'].mean().plot(kind='bar')
###End Seat Usage
#####################################################################


# ## Platform Usage##################
### This field doesnt make sense because mac and mobile apple usage are spars

#df1['Windows_USAGE'].describe()
remove_outlier(ll_df2['Windows_USAGE'])['high']


#df1['Mobile_Apple_USAGE'].describe()
remove_outlier(ll_df2['Mobile_Apple_USAGE'])['high']

#df1['MAC_USAGE'].describe()
remove_outlier(ll_df2['MAC_USAGE'])['high']
##End  Platfor usage
##################################################################



# ## Credit Card Issuing Organization
##################################################################
ll_df2['CARD_BRAND'] = ll_df2['CARD_BRAND'].fillna('Other')


ll_df2['CARD_BRAND']=ll_df2['CARD_BRAND'].apply(lambda x: 'Other' if x in ('PIN ONLY','JCB','DINERS CLUB','INTERAC','INTERNATIONAL MAESTRO') else x)

ll_df2.groupby(['CARD_BRAND'])['LL_FL'].mean()

ll_df2.groupby(['CARD_BRAND'])['LL_FL'].mean().plot(kind='bar')
###end Credit Card
#################################################################




# ## Email Interaction##############################################

ll_df2['NUM_SENT'] = ll_df2['NUM_SENT'].fillna(0)

### very few users have 6 emails so cap at 5
ll_df2['NUM_SENT'] = ll_df2['NUM_SENT'].apply(lambda x: 5 if x > 5 else x)

##mail opened
ll_df2['NUM_OPENED'] = ll_df2['NUM_OPENED'].fillna(0)


##if mail open without sending mail, make it negative, since this is not possibble
ll_df2['NUM_OPENED'].loc[ll_df2.NUM_SENT == 0] = -1


ll_df2['NUM_OPENED'].describe()

ll_df2['NUM_OPENED'].value_counts()

### very few users have 6 emails so cap at 5
ll_df2['NUM_OPENED'] = ll_df2['NUM_OPENED'].apply(lambda x: 5 if x > 5 else x)

##pd.DataFrame(df1['LL_FL'].groupby(df1['NUM_OPENED']).mean().reset_index(name='mean'))
ll_df2.groupby(['NUM_OPENED'])['LL_FL'].mean()

ll_df2.groupby(['NUM_OPENED'])['LL_FL'].mean().plot(kind='bar')


ll_df2['NUM_OPENED'].value_counts()

### Store the value of the column
### df1['NUM_CLICKED1'] = df1['NUM_CLICKED'] 

ll_df2['NUM_CLICKED'] = ll_df2['NUM_CLICKED'].fillna(0)


## -1 for not sent any email
## -2 for not opened any email
ll_df2['NUM_CLICKED'].loc[ll_df2.NUM_SENT == 0] = -1
ll_df2['NUM_CLICKED'].loc[ll_df2.NUM_OPENED == 0] = -2

ll_df2.NUM_CLICKED.value_counts()

#df1[['NUM_CLICKED','NUM_OPENED']].loc[(df1.NUM_CLICKED>-1) & (df1.NUM_OPENED>-1) & (df1.NUM_CLICKED>df1.NUM_OPENED)].head()

### correct rows where clicked > open
ll_df2['NUM_CLICKED'][(ll_df2.NUM_CLICKED>-1) & (ll_df2.NUM_OPENED>-1) & (ll_df2.NUM_CLICKED>ll_df2.NUM_OPENED)] = ll_df2.NUM_OPENED


### very few users have 5 emails so cap at 4, not needed since nobody clicked on all 6 emails
ll_df2['NUM_CLICKED'] = ll_df2['NUM_CLICKED'].apply(lambda x: 4 if x > 4 else x)
### Bucket Variables for exploration
## no emails sent, no emails opened, 
ll_df2['OPEN_RATE'] = -1
ll_df2['OPEN_RATE'].loc[ll_df2.NUM_OPENED>-1] = ll_df2.NUM_OPENED/ll_df2.NUM_SENT

ll_df2['OPEN_RATE'].describe()


ll_df2['OPEN_RATE_BUCKET']=pd.cut(ll_df2['OPEN_RATE'],
                             bins = [-10,-1,0,0.25,0.50,0.75,1],
                             labels = ['No Emails', 'None Opened', '<25%', '25-50%','50-75%','>75%'])


##pd.DataFrame(df1['LL_FL'].groupby(df1['OPEN_RATE_BUCKET']).mean().reset_index(name='mean'))

ll_df2.groupby(['OPEN_RATE_BUCKET'])['LL_FL'].mean()

ll_df2.groupby(['OPEN_RATE_BUCKET'])['LL_FL'].mean().plot(kind='bar')


### Bucket Click Variables for exploration
## no emails sent -2, no emails opened -1, 
ll_df2['CLICK_RATE'] = -2
ll_df2['CLICK_RATE'].loc[ll_df2.NUM_OPENED == 0] = -1
ll_df2['CLICK_RATE'].loc[ll_df2.NUM_OPENED>0] = ll_df2.NUM_CLICKED/ll_df2.NUM_OPENED


ll_df2['CLICK_RATE_BUCKET']=pd.cut(ll_df2['CLICK_RATE'],
                             bins = [-3,-2,-1,0,0.50,1],
                             labels = ['No Emails', 'None Opened','None Clicked' , '<50%','>50%'])



ll_df2.groupby(['CLICK_RATE_BUCKET'])['LL_FL'].mean()

ll_df2.groupby(['CLICK_RATE_BUCKET'])['LL_FL'].mean().plot(kind='bar')
##End enailk Interaction
########################################################################





# ## PIF Interaction###################

ll_df2['NUM_DISP'] = ll_df2['TOTAL_DSPL_FL']


ll_df2['NUM_DISP'].isnull().sum()

ll_df2['NUM_DISP'] = ll_df2['NUM_DISP'].fillna(0)


ll_df2['NUM_DISP'].value_counts()

#pd.DataFrame(df1['LL_FL'].groupby(df1['NUM_DISP']).mean().reset_index(name='mean'))

ll_df2['NUM_PIF_CLK'] = ll_df2['TOTAL_CLK_A']


ll_df2['NUM_PIF_CLK'] = ll_df2['NUM_PIF_CLK'].fillna(0)

### -1 means not displayed a PIF
ll_df2['NUM_PIF_CLK'].loc[ll_df2.NUM_DISP == 0] = -1

ll_df2['NUM_PIF_CLK'].value_counts()

##pif click rate
ll_df2['PIF_CLK_RT'] = -1



ll_df2['PIF_CLK_RT'].loc[ll_df2.NUM_DISP>0] = ll_df2.NUM_PIF_CLK/ll_df2.NUM_DISP

ll_df2['PIF_CLK_RT'].describe()

ll_df2['PIF_CLICK_RATE_BUCKET']=pd.cut(ll_df2['PIF_CLK_RT'],
                             bins = [-2,-1,0,0.25,1],
                             labels = ['No Disp','None Clicked' , '<25%','>25%'])


ll_df2.groupby(['PIF_CLICK_RATE_BUCKET'])['LL_FL'].mean()

ll_df2.groupby(['PIF_CLICK_RATE_BUCKET'])['LL_FL'].mean().plot(kind='bar')
##End PIF##
##################################


# ## TOUCHPOINT INTERACTION#################

ll_df2['TOTAL_TP'] = ll_df2['NUM_SENT'] + ll_df2['NUM_DISP']


ll_df2['TOTAL_TP_INT'] = ll_df2['NUM_PIF_CLK'] + ll_df2['NUM_CLICKED']


ll_df2['TOTAL_TP_INT'].loc[ll_df2.TOTAL_TP<1] = -4


ll_df2['TOTAL_TP_INT'].value_counts()


ll_df2['TOTAL_TP'].value_counts()


ll_df2['TOT_INTR_BUCKET']=pd.cut(ll_df2['TOTAL_TP_INT'],
                             bins = [-6,-4,0,1,10],
                             labels = ['No TP','None Clk' , '1','>1'])


ll_df2['TOT_INTR_BUCKET'].value_counts()

ll_df2.groupby(['TOT_INTR_BUCKET'])['LL_FL'].mean()

ll_df2.groupby(['TOT_INTR_BUCKET'])['LL_FL'].mean().plot(kind='bar')
##End TOUCHPOINT
###############################################################################


ll_df2.head()


# ## Age ##########################################

ll_df2.age_bucket.isnull().sum()



ll_df2.age_bucket.value_counts()


ll_df2['age_bucket'] = ll_df2.age_bucket.fillna('unknown')


### rename the levels in age
def label_age(x):
   # if x == 'IDP_F':
   #     return 'IDP_F'
    if x == '56-65':
        return '56-65'
    elif x == '46-55':
        return '46-55'
    elif x == '66-75':
        return '66-75'
    elif x == 'age: >75':
        return '>75'
    elif x == '36-45':
        return '36-45'  
    elif x == '26-35':
        return '26-35'
    elif x == '<26':
        return '<26'
    else:
        return 'unknown'


ll_df2['age_bucket1'] = ll_df2['age_bucket'].apply(label_age)
##End Age 
##############################################################


# ## Merkle_Adjusted_Wealth_Rating_AU003 
# A household level ranking of wealth access from highest to lowest adjusted relative to the cost of living at each state intersected with MSAs.  Ranking from 00-19 with 19 representing the highest wealth.

ll_df2.Merkle_Adjusted_Wealth_Rating_AU003.isnull().sum()

ll_df2['Merkle_Adjusted_Wealth_Rating_AU003'] = ll_df2.Merkle_Adjusted_Wealth_Rating_AU003.fillna('unknown')


ll_df2.Merkle_Adjusted_Wealth_Rating_AU003.dtype


### rename the levels in current_sub_channel
def label_adj_wealth2(x):
    if x == '00' or x == '01' or x == '02' or x == '03':
        return '0-3'
 
    elif x == '04' or x == '05' or x == '06' or x == '07':
        return '4-7'
 
    elif x == '08' or x == '09' or x == '10' or x == '11':
        return '8-11'
     
    elif x == '12' or x == '13' or x == '14' or x== '15':
        return '12-15'
     
    elif x == '16' or x == '17' or x == '18' or x == '19':
        return '16-19'
    
    else:
        return 'unknown'

ll_df2['Merkle_Adjusted_Wealth_Rating_AU003'] = ll_df2['Merkle_Adjusted_Wealth_Rating_AU003'].astype(str)

ll_df2['Merkle_Adjusted_Wealth_Rating'] = ll_df2['Merkle_Adjusted_Wealth_Rating_AU003'].apply(label_adj_wealth2)


ll_df2.Merkle_Adjusted_Wealth_Rating.value_counts()

##pd.DataFrame(df1['LL_FL'].groupby(df1['Merkle_Adjusted_Wealth_Rating'] ).mean().reset_index(name='mean'))

ll_df2.groupby(['Merkle_Adjusted_Wealth_Rating'])['LL_FL'].mean()

ll_df2.groupby(['Merkle_Adjusted_Wealth_Rating'])['LL_FL'].mean().plot(kind='bar')





# ## Wealth Predictor Score (Wealth_Predictor_ScorAN127)
# Net Worth Predictor Score (in $'000) - Version 3  Wealth Predictor Score is a predictor of net worth at the household level. It is developed using a combination of data including Liquid Asset Predictor, Home Value and many other demographic and behavioral data elements.  Ranges from -500 to 9999999. (Household)


ll_df2.Wealth_Predictor_ScorAN127.isnull().sum() ##1118768


ll_df2.Wealth_Predictor_ScorAN127 = ll_df2.Wealth_Predictor_ScorAN127.fillna(-100000)


ll_df2.Wealth_Predictor_ScorAN127.describe()



ll_df2.Wealth_Predictor_ScorAN127.dtype


ll_df2.Wealth_Predictor_ScorAN127[ll_df2.Wealth_Predictor_ScorAN127== '\\N'] = -100001


ll_df2.Wealth_Predictor_ScorAN127[ll_df2.Wealth_Predictor_ScorAN127== '\\\\N'] = -100001

ll_df2.Wealth_Predictor_ScorAN127_1 = ll_df2['Wealth_Predictor_ScorAN127'].astype(str).astype(int)

ll_df2.Wealth_Predictor_ScorAN127_1.describe()


ll_df2.Wealth_Predictor_ScorAN127_1.max()


ll_df2.Wealth_Predictor_ScorAN127_1[ll_df2.Wealth_Predictor_ScorAN127_1>-100000].min()


q1 = ll_df2.Wealth_Predictor_ScorAN127_1[ll_df2.Wealth_Predictor_ScorAN127_1>-100000].quantile(q=[0.25, 0.5, 0.75])



ll_df2['Wealth_Predictor_Scor'] = pd.cut(ll_df2.Wealth_Predictor_ScorAN127_1,
                                  bins = [-200000,-90000,75,150,300,750,29000],
                                  labels = ['unknown','<75K','75K-150K','150K-300K','300K-750K','>750K'])


ll_df2.Wealth_Predictor_Scor.value_counts()



##pd.DataFrame(df1['LL_FL'].groupby(df1['Wealth_Predictor_Scor'] ).mean().reset_index(name='mean'))

ll_df2.groupby(['Wealth_Predictor_Scor'])['LL_FL'].mean()

ll_df2.groupby(['Wealth_Predictor_Scor'])['LL_FL'].mean().plot(kind='bar')






# ## Income


ll_df2['income'].isnull().sum()


ll_df2['income'] = ll_df2.income.fillna('unknown')


### rename the levels in age
def label_inc(x):
    if x == '$1,000 - $14,999' or x == '$15,000 - $24,999' or x == '$25,000 - $34,999' or x == '$35,000 - $49,999':
        return '<50K'
    
    elif x == '$50,000 - $74,999' or x == '$75,000 - $99,999':
        return '50K-100K'  
    
    elif x == '$100,000 - $124,999' or x == '$125,000 - $149,999':
        return '100K-150K'
    elif x == '$150,000 - $174,999' or x == '$175,000 - $199,999':
        return '150K-200K'
    
    elif x == '$200,000 - $249,999' or x == '$250,000+':
        return '>200K'
    else:
        return 'unknown'


ll_df2['income1']=ll_df2['income'].apply(label_inc)


##pd.DataFrame(df1['LL_FL'].groupby(df1['income1']).mean().reset_index(name='mean'))



# ## Credit_Score_Range_VW107  
# A = 800+ B = 750-799 C = 700-749
# D = 650-699 E = 600-649 F = 550-599
# G = 500-549 H = 499& Less
ll_df2.Credit_Score_Range_VW107.isnull().sum()

#df1.Credit_Score_Range.value_counts()

ll_df2['Credit_Score_Range'] = ll_df2.Credit_Score_Range_VW107.fillna('unknown')


ll_df2['Credit_Score_Range'][ll_df2.Credit_Score_Range=='\\N'] = 'unknown'

ll_df2['Credit_Score_Range'][ll_df2.Credit_Score_Range=='\\\\N'] = 'unknown'



##pd.DataFrame(df1['LL_FL'].groupby(df1['Credit_Score_Range'] ).mean().reset_index(name='mean')) 

ll_df2.groupby(['Credit_Score_Range'])['LL_FL'].mean()

ll_df2.groupby(['Credit_Score_Range'])['LL_FL'].mean().plot(kind='bar')






# ## Facebook Influencer

ll_df2.Facebook_Influencer_AN131.isnull().sum()



ll_df2.Facebook_Influencer_AN131 = ll_df2.Facebook_Influencer_AN131.fillna('-1')



ll_df2.Facebook_Influencer_AN131[ll_df2.Facebook_Influencer_AN131=='\\\\N'] = '-1'



ll_df2.Facebook_Influencer_AN131[ll_df2.Facebook_Influencer_AN131=='\\N'] = '-1'


ll_df2.Facebook_Influencer_AN131.value_counts()


ll_df2['Facebook_Influencer'] = ll_df2.Facebook_Influencer_AN131


##pd.DataFrame(df1['LL_FL'].groupby(df1['Facebook_Influencer'] ).mean().reset_index(name='mean'))

ll_df2.groupby(['Facebook_Influencer'])['LL_FL'].mean()
ll_df2.groupby(['Facebook_Influencer'])['LL_FL'].mean().plot(kind='bar')



# ## Children#############

ll_df2.Children_DS907.isnull().sum()

ll_df2.Children_DS907 = ll_df2.Children_DS907.fillna('unknown')

ll_df2.Children_DS907[ll_df2.Children_DS907=='\\N'] = 'unknown'

ll_df2.Children_DS907[ll_df2.Children_DS907=='\\\\N'] = 'unknown'

ll_df2.Children_DS907.value_counts()
ll_df2['children'] = ll_df2.Children_DS907

##pd.DataFrame(df1['LL_FL'].groupby(df1['children'] ).mean().reset_index(name='mean'))
ll_df2.groupby(['children'])['LL_FL'].mean()
ll_df2.groupby(['children'])['LL_FL'].mean().plot(kind='bar')


# ## Marital Status ###

ll_df2['marital_status'] = ll_df2['marital_status'].fillna('unknown')
ll_df2['marital_status'].value_counts()

ll_df2.groupby(['marital_status'])['LL_FL'].mean()
ll_df2.groupby(['marital_status'])['LL_FL'].mean().plot(kind='bar')



# ## Homeowner


ll_df2['Homeowner'] = ll_df2.Homeowner_DS921.fillna('unknown')


ll_df2['Homeowner'][ll_df2.Homeowner=='\\N'] = 'unknown'

ll_df2['Homeowner'][ll_df2.Homeowner=='\\\\N'] = 'unknown'

ll_df2.groupby(['Homeowner'])['LL_FL'].mean()
ll_df2.groupby(['Homeowner'])['LL_FL'].mean().plot(kind='bar')



# ## Gender


ll_df2['gender'] = ll_df2.gender.fillna('unknown')


ll_df2['gender'].value_counts()

ll_df2.groupby('gender')['LL_FL'].mean()
ll_df2.groupby('gender')['LL_FL'].mean().plot(kind='bar')

# ## Education############
ll_df2['education'].value_counts()
ll_df2['education'].isnull().sum()
ll_df2['education'] = ll_df2['education'].fillna('unknown')

##pd.DataFrame(df1['LL_FL'].groupby(df1['education'] ).mean().reset_index(name='mean')) 

ll_df2.groupby('education').agg({'LL_FL':'mean', 'education':'count'})
ll_df2.groupby('education')['LL_FL'].mean().plot(kind='bar')



# ## Number of people in household  (Number_Persons_Living_Unit_MS947 )

ll_df2.Number_Persons_Living_Unit_MS947.value_counts()
ll_df2.Number_Persons_Living_Unit_MS947[ll_df2.Number_Persons_Living_Unit_MS947.isin(['\\N','\\\\N'])] = '-1'

ll_df2.Number_Persons_Living_Unit_MS947.value_counts()



ll_df2.Number_Persons_Living_Unit_MS947.isnull().sum()
ll_df2.Number_Persons_Living_Unit_MS947 = ll_df2.Number_Persons_Living_Unit_MS947.fillna('-1')


ll_df2.Number_Persons_Living_Unit = ll_df2.Number_Persons_Living_Unit_MS947.astype(str).astype(int)


ll_df2['Num_Per_In_Unt'] = pd.cut(ll_df2.Number_Persons_Living_Unit,
                           bins = [-2,-1,1,2,3,10],
                           labels = ['Missing','1','2','3','>3'])

##pd.DataFrame(df1['LL_FL'].groupby(df1['Num_Per_In_Unt'] ).mean().reset_index(name='mean')) 

ll_df2.groupby('Num_Per_In_Unt').agg({'LL_FL':'mean', 'Num_Per_In_Unt':'count'})
ll_df2.groupby('Num_Per_In_Unt')['LL_FL'].mean().plot(kind='bar')

##optout AR flag
ll_df2.OPT_OUT_VALUE_AR_HIST = ll_df2.OPT_OUT_VALUE_AR_HIST.fillna('missing')
ll_df2.groupby('OPT_OUT_VALUE_AR_HIST').agg({'LL_FL':'mean', 'OPT_OUT_VALUE_AR_HIST':'count'})
ll_df2.groupby('OPT_OUT_VALUE_AR_HIST')['LL_FL'].mean().plot(kind='bar')


##optout AR flag
ll_df2.CARD_TYPE = ll_df2.CARD_TYPE.fillna('unknown')
ll_df2.groupby('CARD_TYPE').agg({'LL_FL':'mean', 'CARD_TYPE':'count'})
ll_df2.groupby('CARD_TYPE')['LL_FL'].mean().plot(kind='bar')


##user agent 
ll_df2.Device_name = ll_df2.Device_name.fillna('NoVisit')
ll_df2.is_mobile = ll_df2.is_mobile.fillna('NoVisit')
ll_df2.OS_Family_catogary = ll_df2.OS_Family_catogary.fillna('NoVisit')
ll_df2.Browser_Family_catogary = ll_df2.Browser_Family_catogary.fillna('NoVisit')

# ## Save full dataset for exploration
### save the datafrme to csv
ll_df2.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/ll_cl_0604.csv', encoding='utf-8', index=False)


# ## Save dataset for modeling 

## select features
nm = [
'ACCOUNT_GUID',
'PSN',   
'DAYS_TO_EXPIRE_BIN',
'PSN_CHANNEL_N',
'CURRENT_AR_FLAG',
'AR_UPGRADE_PURCHASE',
'ACTIVE_PRODUCTS_PROC1',
'OPT_OUT_VALUE_AR_HIST',
'TENURE_YR',
'TENURE_D',
'SEAT_USAGE1',
'CARD_BRAND',
'PIF_CLICK_RATE_BUCKET',
'NUM_PIF_CLK',
'NUM_DISP',
'CLICK_RATE_BUCKET',
'OPEN_RATE_BUCKET',
'NUM_CLICKED',
'NUM_OPENED',
'NUM_SENT',
'TOT_INTR_BUCKET',
'age_bucket1',
'Merkle_Adjusted_Wealth_Rating',
'Wealth_Predictor_Scor',
'income1','Credit_Score_Range',
'Facebook_Influencer',
'children', 'marital_status',
 'Homeowner', 'gender','education',
 'Num_Per_In_Unt',  
'Landing_web_site_catogary', 
## 'Device_name',
 'is_mobile',
 'Browser_Family_catogary',
 'OS_Family_catogary',
'LL_FL']

dfs = ll_df2[nm]


### Remove 'Ignore' category from subscription channel
dfm = remLev(dfs,'PSN_CHANNEL_N','Ignore')


###Take out rows with missing DAYS_TO_EXPIRE_BIN

dfm = dfm[dfm.DAYS_TO_EXPIRE_BIN.notnull()]


### save the dataframe 
dfm.to_pickle('/Users/razzak_lebbai/Desktop/propensity/data/ll_mod_0604.pkl')

