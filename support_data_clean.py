#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:21:21 2018

@author: razzak_lebbai
"""


import pandas as pd
import numpy as np

##from functions_propencity import *
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
   
def convergerCNTType(x):
    
    
    contact_type={'phone':'phone',
                     'chat':'chat',
                     'Phone':'phone',
                     'email':'email',
                     'Email':'email',
                     'Mailbox':'Mailbox',
                     'Letter':'Letter',
                     'Social Support':'Social Support',
                     'Social Su':'Social Support',
                     'Support T':'Social Support',
                     'RMA Return':'RMA Return',
                     'RMA Retur':'RMA Return'
                    
                      }
    if x in contact_type.keys() :
        return contact_type[x]
    else:
        return 'other_cnt_type'
   
    
def convergerCaseType(x):
    
    
    case_type={'Product Support':'Product Support',
                     'Computer Tune Up':'Computer Tune Up',
                     'VSSG - Virus Removal Assurance':'Virus Removal Assurance',
                     'Norton Core':'Norton Core',
                     'UHD':'UHD',
                     'VSS - Spyware and Virus Removal':'Virus Removal Assurance',
                     'VSSA - Virus Removal Assurance':'Virus Removal Assurance',
                     'VSSB - Virus Removal Assurance':'Virus Removal Assurance',
                     'UHD - Annual':'UHD',
                     'Virus Removal Assurance (VSSa)':'Virus Removal Assurance',
                     'UHD - Monthly':'UHD',
                     'Support':'Product Support',
                     'Spyware and Virus Removal (VSS)':'Virus Removal Assurance'
                    
                      }
    if x in case_type.keys() :
        return case_type[x]
    else:
        return 'other_case_type'
    

def convergerOutBType(x):
    
    
    OutB_type={'Inbound':'Inbound',
               'Outbound':'Outbound',
                     'Inbound - Phone':'Inbound',
                     'Outbound - Phone':'Outbound',
                       }
    if x in OutB_type.keys() :
        return OutB_type[x]
    else:
        return 'other_OutB_type'
    

def countDummy(df, NORTON_ACCT_GUID='NORTON_ACCT_GUID' ):
    count_dummy_df = pd.DataFrame(df.NORTON_ACCT_GUID.unique(), columns=['NORTON_ACCT_GUID'])
    columnsDummy = list(df.select_dtypes(include=['category','uint8']))
    for col in columnsDummy:
        temp_count= df.groupby(NORTON_ACCT_GUID)[col].count().reset_index()
        count_dummy_df = count_dummy_df.merge(temp_count, on='NORTON_ACCT_GUID', how='inner')
        
        
        
    return count_dummy_df
    
    
    

##usewr_agent 

support_data_df=pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/support_data.csv')

count_cases = support_data_df.groupby('NORTON_ACCT_GUID')['CASE_ID'].count().reset_index()

##calculate contact type 
contact_type = support_data_df[['NORTON_ACCT_GUID','CNTCT_TYPE_CD']]
contact_type['CNTCT_TYPE_CD_mod'] = contact_type.CNTCT_TYPE_CD.apply(convergerCNTType)
contact_type = pd.concat([contact_type, pd.get_dummies(contact_type.CNTCT_TYPE_CD_mod).rename(columns=lambda x: 'CNTCT_TYPE_CD_mod' + '_' + str(x))], axis=1)
count_contact_type = countDummy(contact_type)
##calculate type of cases######

case_type = support_data_df[['NORTON_ACCT_GUID','CASE_TYPE_CD']]
case_type['CASE_TYPE_CD_mod'] = case_type.CASE_TYPE_CD.apply(convergerCaseType)
case_type = pd.concat([case_type, pd.get_dummies(case_type.CASE_TYPE_CD_mod).rename(columns=lambda x: 'CASE_TYPE_CD_mod' + '_' + str(x))], axis=1)
count_case_type = countDummy(case_type)


##calculate outbount cll type
outb_type = support_data_df[['NORTON_ACCT_GUID','OUTB_CALL_TYPE_CD']]
outb_type['OUTB_CALL_TYPE_CD_mod'] = outb_type.OUTB_CALL_TYPE_CD.apply(convergerOutBType)
outb_type = pd.concat([outb_type, pd.get_dummies(outb_type.OUTB_CALL_TYPE_CD_mod).rename(columns=lambda x: 'OUTB_CALL_TYPE_CD_mod' + '_' + str(x))], axis=1)
count_outb_type = countDummy(outb_type)


##IS_AUTO_CRT_CASE_IN######
auto_case_create = support_data_df[['NORTON_ACCT_GUID','IS_AUTO_CRT_CASE_IN']]
##auto_case_create['IS_AUTO_CRT_CASE_IN_mod'] = auto_case_create.IS_AUTO_CRT_CASE_IN_mod.apply(convergerOutBType)
auto_case_create = pd.concat([auto_case_create, pd.get_dummies(auto_case_create.IS_AUTO_CRT_CASE_IN).rename(columns=lambda x: 'IS_AUTO_CRT_CASE_IN' + '_' + str(x))], axis=1)
count_auto_case_create = countDummy(auto_case_create)

##cobine all
support_count = count_cases.merge(count_contact_type, on='NORTON_ACCT_GUID', how='inner').\
merge(count_case_type, on='NORTON_ACCT_GUID', how='inner').\
merge(count_outb_type, on='NORTON_ACCT_GUID', how='inner').\
merge(count_auto_case_create, on='NORTON_ACCT_GUID', how='inner')
support_count = support_count.rename(columns={'NORTON_ACCT_GUID':'ACCOUNT_GUID'})


##LL matched customers
ll_df =  pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/data_v4.csv',  nrows=None, usecols=['ACCOUNT_GUID', 'LL_FL'])

##merger withLL data

ll_support_df  = ll_df.merge(support_count, on='ACCOUNT_GUID', how='left')

matched_percentage = ll_support_df.CASE_ID.notnull().sum()/ll_support_df.shape[0]
sucess = ll_support_df[ll_support_df.CASE_ID.notnull()==True]['LL_FL'].value_counts()

ll_support_clean_df = ll_support_df.fillna(0)

ll_support_clean_df.to_csv('/Users/razzak_lebbai/Desktop/propensity/data/ll_support_clean.csv', index=False)