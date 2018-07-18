#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:28:24 2018

@author: razzak_lebbai
"""
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from user_agents import parse
from datetime import timedelta
##country code list 
## this needs to check landing page with contry code
cn_code= pd.read_csv('/Users/razzak_lebbai/Desktop/omniture/data/country_code.csv')
cncode_list = cn_code.EVR_27_PRP_2_SITE_CTRY_CD.tolist()



###convert datetime type
def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)
    


def ActiveCustomer(visit_date, expiration_date):
    
    if (expiration_date >= visit_date + relativedelta(days=-60))   and expiration_date <visit_date + relativedelta(days=365):
        
        return 1
    else:
        return 0
    
    


def ActiveCustomer60Days(visit_date, expiration_date):
    
    if (expiration_date >= visit_date + relativedelta(days=-60))   and expiration_date <= visit_date + relativedelta(days=+60):
        
        return 1
    else:
        return 0
    
    



def WinBackCustomer(visit_date, expiration_date):
    
    if (expiration_date < visit_date + relativedelta(days=-60)):
        
        return 1
    else:
        return 0
    
    

   
##customer expired 60 or less 
def ActiveCustomer60Minus(visit_date, expiration_date):
    
    if (expiration_date >= visit_date + relativedelta(days=-60))   and expiration_date <= visit_date:
        
        return 1
    else:
        return 0
    
 
      
   
def CustomerSelectionWithExpDate(visit_date, expiration_date, days_before_exp, days_after_exp):
    
    if (expiration_date >= visit_date + relativedelta(days=days_before_exp))   and expiration_date <= visit_date + relativedelta(days=days_after_exp):
        
        return 1
    else:
        return 0
    
 
   
    
    

def convergerBrowser(x):
    
    browser = {
            'Chrome':'Chrome',
            'Chrome Mobile':'Chrome',
            'Chrome Mobile iOS':'Chrome',
            'Edge':'IE',
            'IE':'IE',
            'IE Mobile':'IE',
            'Edge Mobile':'IE',
            'Firefox':'Firefox',
            'Firefox Mobile':'Firefox',
            'Mobile Safari':'Safari',
            'Safari':'Safari'
            }
    if x in browser.keys() :
        return browser[x]
    else:
        return 'other_browser'
    
    
         
    
def convergerOS(x):
    
    OS_fam = {
            'Windows 10':'Windows',
            'Windows 7':'Windows',
            'Windows 8.1':'Windows',
            'Windows Vista':'Windows',
            'Windows 8':'Windows',
            'Windows XP':'Windows',
            'Windows RT 8.1':'Windows',
            'Windows 2000':'Windows',
            'Windows RT':'Windows',
            'Windows NT 4.0 ':'Windows',
            'Windows':'Windows',
            'Mac OS X':'MacOSX',
            'iOS':'iOS',
            'Android':'Android',
            'Linux':'Linux',
            'Ubuntu':'Linux',
            'OpenBSD':'Linux',
            'Fedora':'Linux',
            'Chrome OS':'ChromeOS'
                    }
    if x in OS_fam.keys() :
        return OS_fam[x]
    else:
        return 'other_OS'
    
    


##User agent parser

## iPhone's user agent string
#ua_string = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_1 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9B179 Safari/7534.48.3'
#user_agent = parse(ua_string)
#
## Accessing user agent's browser attributes
#user_agent.browser # returns Browser(family=u'Mobile Safari', version=(5, 1), version_string='5.1')
#user_agent.browser.family # returns 'Mobile Safari'
#user_agent.browser.version # returns (5, 1)
#user_agent.browser.version_string # returns '5.1'
#
## Accessing user agent's operating system properties
#user_agent.os # returns OperatingSystem(family=u'iOS', version=(5, 1), version_string='5.1')
#user_agent.os.family # returns 'iOS'
#user_agent.os.version # returns (5, 1)
#user_agent.os.version_string # returns '5.1'
#
## Accessing user agent's device properties
#user_agent.device # returns Device(family=u'iPhone', brand=u'Apple', model=u'iPhone')
#user_agent.device.family # returns 'iPhone'
#user_agent.device.brand # returns 'Apple'
#user_agent.device.model # returns 'iPhone'

def parse_useragent(x):
    ua_string = x
    user_agent = parse(ua_string)
    return user_agent

    
    



###@#
##creatre session ID using  time stamp
    
    
def LandingPage(x):
    return x[x.HIT_DTTM_GMT==x.HIT_DTTM_GMT.min()]['VISIT_STRT_PG_URL_NM'].unique()[0]


##get the landing page from VISIT_STRT_PG_URL_NM
def LandingPageFinal(df, session_id='Visit_Session_Id'):
    df['Landing_page'] = df.groupby(session_id)['VISIT_STRT_PG_URL_NM_test'].transform(lambda x: x.unique()[0])
    
    ##landing_df = landing_df.reset_index(name='Landing_Page')
    
    
 ##parse url
from urllib.parse import urlparse 
def ParseUrl(url):
    return urlparse(url)



##recalculate landing page
def FillVISIT_STRT_PG_URL_NM(x, landing_page='VISIT_STRT_PG_URL_NM_test'):
#    min_value = x.HIT_DTTM_GMT.min()
    first_page_visit = x[x.HIT_DTTM_GMT==x.HIT_DTTM_GMT.min()]['PG_URL_NM']
    x[landing_page] = x[landing_page].fillna(value=first_page_visit.values[0])
    
    return x
    

def FillVISIT_STRT_PG_URL_NM1(x, landing_page='VISIT_STRT_PG_URL_NM_test'):
#    min_value = x.HIT_DTTM_GMT.min()
    first_page_visit = x[x.HIT_DTTM_GMT==x.HIT_DTTM_GMT.min()]['PG_URL_NM']
    ##x[landing_page] = x[landing_page].fillna(value=first_page_visit.values[0])
    return first_page_visit.values[0]

    
    ##print(x.VISIT_STRT_PG_URL_NM)
    
 

##recalculate landing page
def FillVISIT_STRT_PG_URL_NM2(x, landing_page='VISIT_STRT_PG_URL_NM_test'):
#    min_value = x.HIT_DTTM_GMT.min()
    x= x.sort_values('HIT_DTTM_GMT')
    x[landing_page] = x[landing_page].fillna(value=x[landing_page][x.index[0]])
    
    return x
    
##get dummies() for browsers
def find_dummies(df, column):
    
    dummy_df= df[column].str.get_dummies()

    dummy_df.columns = ['is_' + col for col in dummy_df.columns]
    
    df[dummy_df.columns.tolist()]=dummy_df
    
    
    
    
##get dummies() for browsers
def find_dummies_last(df, column):
    
    dummy_df= df[column].str.get_dummies()

    dummy_df.columns = ['last_' + col for col in dummy_df.columns]
    
    df[dummy_df.columns.tolist()]=dummy_df
    ##return dummy_df
    
#  
###count number is_colums for each colums
#def Count_feature_PSN(df, col, session_id='Estore_Session_Id'):
#    ##is_cols = [col for col in df.columns if 'is_' in col]
#    nodup_df =df.drop_duplicates(subset=[session_id, 'INCM_PSN_NUM', col])
#    
#    number_df = nodup_df.groupby('INCM_PSN_NUM')[col].sum()
#    number_df =  number_df.reset_index(name='total_'+col)
#    
#    return number_df
#
#
##this function calculate both number of is_col per pSNA and avarage sum of is_col per session
def Ave_sum_feature_count_per_visit(df, col, session_id='Estore_Session_Id'):
    nodup_df =df.drop_duplicates(subset=[session_id, 'INCM_PSN_NUM', col])
    
    number_df = nodup_df.groupby('INCM_PSN_NUM').agg({col:np.sum, session_id:'count'}).reset_index().rename(columns={col:'total_'+col, session_id:'total_session'})
    number_df['ave_'+col+'_per_session']=number_df['total_'+col]/number_df['total_session']
    
    del number_df['total_session']
    return number_df
    
    
    
    
    
#    
#   ## test2.groupby('INCM_PSN_NUM').agg({'is_IE':np.sum, 'Estore_Session_Id':'count'}).reset_index()
###avarage clicks persession per PSN
#def Ave_feature_count_per_visit(df, col, session_id='Visit_Session_Id'):
#    
#    count_per_visit = df.groupby(session_id)[col].sum()
#    count_per_visit = count_per_visit.reset_index(name='count_web_visit')
#    
#    
#    count_per_visit[['Date', 'INCM_PSN_NUM']]= count_per_visit[session_id].apply(lambda x: pd.Series(str(x).split('-')))
#    ##number of clicks
#    df_count_per_visit_final = count_per_visit.groupby('INCM_PSN_NUM')['count_web_visit'].sum()/count_per_visit.groupby('INCM_PSN_NUM')['count_web_visit'].count()
#    
#    return df_count_per_visit_final
#    
#

    
 
##catogarise landing pages  
def Landing_web_site_parse(x, Landing_web_site_df,  nortondotcom=cncode_list):
    ##Landing_web_site_df = pd.DataFrame(df.Landing_web_site.unique(), columns=['page'])
    Landing_web_site={
            'renew.norton.com':'renew.norton.com',\
            'analytics-login.norton.com':'analytics-login.norton.com',
            'my.norton.com':'my.norton.com'
            }
    ##add all type byu.xx as buy.norton.com
    for site in Landing_web_site_df[Landing_web_site_df.page.str.contains('buy')==True]['page'].tolist():
        Landing_web_site[site]='buy.norton.com'
    
    ##get norton.com
    nortondotcom = [k+'.norton.com' for k in cncode_list]
    ##list the common set 
    available_nortondotcom = list(set(Landing_web_site_df.page.tolist()) & set(nortondotcom))
    ##now adsd all norto.com to the dictionary 
    
    for nordcom in available_nortondotcom:
        Landing_web_site[nordcom]='country.norton.com'
    
    ##add all type support.xx as buy.norton.com
    for site in Landing_web_site_df[Landing_web_site_df.page.str.contains('support')==True]['page'].tolist():
        Landing_web_site[site]='support.norton.com'
        
        
    
    if x in Landing_web_site.keys() :
        return Landing_web_site[x]
    else:
        return 'other_landing_page'
    

 
      
##catogarise landing pages  
def Landing_web_site_keys(Landing_web_site_df,  nortondotcom=cncode_list):
    ##Landing_web_site_df = pd.DataFrame(df.Landing_web_site.unique(), columns=['page'])
    Landing_web_site={
            'renew.norton.com':'renew.norton.com',\
            'analytics-login.norton.com':'analytics-login.norton.com',
            'my.norton.com':'my.norton.com'
            }
    ##add all type byu.xx as buy.norton.com
    for site in Landing_web_site_df[Landing_web_site_df.page.str.contains('buy')==True]['page'].tolist():
        Landing_web_site[site]='buy.norton.com'
    
    ##get norton.com
    nortondotcom = [k+'.norton.com' for k in cncode_list]
    ##list the common set 
    available_nortondotcom = list(set(Landing_web_site_df.page.tolist()) & set(nortondotcom))
    ##now adsd all norto.com to the dictionary 
    
    for nordcom in available_nortondotcom:
        Landing_web_site[nordcom]='country.norton.com'
    
    ##add all type support.xx as buy.norton.com
    for site in Landing_web_site_df[Landing_web_site_df.page.str.contains('support')==True]['page'].tolist():
        Landing_web_site[site]='support.norton.com'
        
        
    
    return Landing_web_site
    

##catogarise landing pages  
def Landing_web_site_parse1(x, Landing_web_site_keys,  nortondotcom=cncode_list):
    if x in Landing_web_site_keys.keys() :
        return Landing_web_site_keys[x]
    else:
        return 'other_landing_page'


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


    
def convergerProd(x):
    
    
    product_family={'NS':'NS',
                     'NS-S':'NS',
                     'NS-SP':'NS',
                     'NIS':'NIS',
                     'NS-BU':'NS-BU',
                     'NOF-PE':'NOF',
                     'NSW-PE':'NSW',
                     'N360':'N360',
                     'N360-MD':'N360',
                     'N360-PE':'N360',
                     'LL_STD_ADD':'LL_STD_ADD',
                     'NAVB':'NAVB',
                     'NAV':'NAV',
                     'NS-BU-STORAGE':'NS-BU-STORAGE'
                      }
    if x in product_family.keys() :
        return product_family[x]
    else:
        return 'other_prod'
    
    
