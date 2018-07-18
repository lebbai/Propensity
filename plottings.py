#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:50:33 2018

@author: razzak_lebbai
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
from pandas.tools.plotting import table
import six 
import numpy as np
  


def BarPlotWithGroupNoAnoSci(ag, kind = 'bar', colormap = cm.Accent, width = 0.95, title=None, path='/Users/razzak_lebbai/junk/test.png'):   
   
    ax= ag.plot(kind=kind, colormap=colormap, width=width, title=title)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    

def BarPlotWithGroupNoAnoNosci(ag, kind = 'bar', colormap = cm.Accent, width = 0.95, title=None,bbox_to_anchor=(0.9,0.9),  path='/Users/razzak_lebbai/junk/test.png'):   
   
    ax= ag.plot(kind=kind, colormap=colormap, width=width, title=title)
#    for p in ax.patches:
#        ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=bbox_to_anchor)
    ##ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def BarPlotWithGroupAnoNosci(ag, kind = 'bar', colormap = cm.Accent, width = 0.95, title=None, path='/Users/razzak_lebbai/junk/test.png'):   
   
    ax= ag.plot(kind=kind, colormap=colormap, width=width, title=title)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,-0.05))
    ##ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
 

def BarPlotWithGroup(ag, kind = 'bar', colormap = cm.Accent, width = 0.95, title=None, path='/Users/razzak_lebbai/junk/test.png'):   
   
    ax= ag.plot(kind=kind, colormap=colormap, width=width, title=title)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
    ##plt.figure(figsize=(16,9))
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
 
    
##plot tables
def plottable(df, path='/Users/razzak_lebbai/junk/test.png'):
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ##ax.patch.set_visible(False)
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    
    table(ax,df, loc='top')  # where df is your data frame
    plt.tight_layout()
    ##plt.show()
    
    plt.savefig(path)  
    plt.show()
    
 

##plot pandas table 
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w', path='/Users/razzak_lebbai/junk/test.png',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='semibold', color='w', size='medium')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    plt.savefig(path)
    plt.show()
    return ax


def BarPlotWithGroupAnoNosci2(ag, ag_total, kind = 'bar', colormap = cm.Accent, width = 0.95, title=None, path='/Users/razzak_lebbai/junk/test.png'):   
   
    ax= ag.plot(kind=kind, colormap=colormap, width=width, title=title)
    for p in range(len(ax.patches)):
        ax.annotate(str(ax.patches[p].get_height()), (ax.patches[p].get_x() * 1.01, ax.patches[p].get_height() * 1.01))
        ax.annotate(str(ag_total.loc[ag_total.index[p]]['total_customer']), (ax.patches[p].get_x() + 0.45*ax.patches[p].get_width(), ax.patches[p].get_height() * 0.8), rotation=90)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,-0.05))
    ##ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def BarPlotWithGroupAnoNosci3(ag, ag_total, kind = 'bar', colormap = cm.Accent, width = 0.95, title=None, path='/Users/razzak_lebbai/junk/test.png'):   
   
    ax= ag.plot(kind=kind, colormap=colormap, width=width, title=title)
    for p in range(len(ax.patches)):
        ##ax.annotate(str(ax.patches[p].get_height()), (ax.patches[p].get_x() * 1.01, ax.patches[p].get_height() * 1.01))
        ax.annotate(str(ag_total.loc[ag_total.index[p]]['total_customer']), (ax.patches[p].get_x() + 0.45*ax.patches[p].get_width(), ax.patches[p].get_height() * 0.8), rotation=90)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,-0.05))
    ##ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
