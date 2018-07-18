
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
##init_notebook_mode(connected=True) 

from plottings import *

import pandas as pd
import numpy as np
import datetime as dt
import warnings
#warnings.filterwarnings(action='once')
warnings.simplefilter('ignore')
from ggplot import *


df_onesvm = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/output/onesvm_final_result_05.csv')
#if mail open without sending mail, make it negative, since this is not possibble
df_onesvm['Pred_value'].loc[df_onesvm['Pred_value'] == -1] = 0


df_rf = pd.read_csv('/Users/razzak_lebbai/Desktop/propensity/data/output/rf_final_result.csv')



##get successfull ones
df_rf_success= df_rf[df_rf.LL_FL==1]
df_onesvm_success= df_onesvm[df_onesvm.LL_FL==1]



##un success oness

##get successfull ones
df_rf_unsuccess= df_rf[df_rf.LL_FL==0]
df_unonesvm_success= df_onesvm[df_onesvm.LL_FL==0]



BarPlotWithGroupAnoNosci((df_rf_success.Pred_value.value_counts()/df_rf_success.shape[0]), title='RF success comparision', path='/Users/razzak_lebbai/Desktop/propensity/src/feature/v2/plots/rf_success.png')

BarPlotWithGroupAnoNosci((df_onesvm_success.Pred_value.value_counts()/df_onesvm_success.shape[0]), title='OneSvm success comparision', path='/Users/razzak_lebbai/Desktop/propensity/src/feature/v2/plots/onesvm_success.png')

BarPlotWithGroupAnoNosci((df_rf_unsuccess.Pred_value.value_counts()/df_rf_unsuccess.shape[0]), title='RF unsuccess comparision', path='/Users/razzak_lebbai/Desktop/propensity/src/feature/v2/plots/rf_unsuccess.png')

BarPlotWithGroupAnoNosci((df_unonesvm_success.Pred_value.value_counts()/df_unonesvm_success.shape[0]), title='OneSvm unsuccess comparision', path='/Users/razzak_lebbai/Desktop/propensity/src/feature/v2/plots/onesvm_unsuccess.png' )


##
df_unonesvm_success[df_unonesvm_success.Pred_value==1].decision_function.describe()

df_unonesvm_success[df_unonesvm_success.Pred_value==0].decision_function.describe()



