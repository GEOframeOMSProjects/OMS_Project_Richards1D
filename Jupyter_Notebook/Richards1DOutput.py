# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:42:31 2018

@author: Niccolo` Tubini and Riccardo Rigon
@license: creative commons 4.0
"""

#from netCDF4_classic import Dataset
from netCDF4 import Dataset


import os

## pandas
import pandas as pd

## numpy
import numpy as np

## plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.ticker as mtick

## to convert unix time to human readable date
import time
import datetime

# Standard imports 
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import BoxSelectTool
from bokeh.models import HoverTool

output_notebook()



def setPlotFeatures(labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):
    
    return [labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1]

#[labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1] = setPlotFeatures(labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1)


def readRichardsOutputNetCDF(fileName):
    
    ## open netCDF file for reading.
    ncfile = Dataset(fileName,'r') 
    print ('*** SUCCESS reading')
    
    print('\n ***FILE INFO:\n')
    print(ncfile)
    
    ## other usefull commands:
    #print (ncfile.dimensions['time'])
    #print (ncfile.file_format)
    #print (ncfile.dimensions.keys())
    #print (ncfile.variables.keys())
    #print (ncfile.variables['psi'])
    
    
    depths = ncfile.variables['depth']
    #print('\n ***DEPTHS INFO:\n')
    #print(depths)
    
    dualDepths = ncfile.variables['dual_depth']
    #print('\n ***DUAL DEPTH INFO:\n')
    #print(dualDepths)
    
    time = ncfile.variables['time']
    #print('\n ***TIME INFO:\n')
    #print(time)
    
    psi = ncfile.variables['psi']
    #print('\n ***PSI INFO:\n')
    #print(psi)

    theta = ncfile.variables['water_heigth']
    #print('\n ***THETA INFO:\n')
    #print(theta)

    iC = ncfile.variables['psiIC']
    #print('\n ***INITIAL CONDITION INFO:\n')
    #print(iC)
    
    velocities = ncfile.variables['velocities']
    #print('\n ***VELOCITIES INFO:\n')
    #print(velocities)

    error = ncfile.variables['error']
    #print('\n ***ERROR INFO:\n')
    #print(error)
    
    topBC = ncfile.variables['topBC']
    #nt('\n ***topBC INFO:\n')
    #print(topBC)
    
    bottomBC = ncfile.variables['bottomBC']
    #print('\n ***bottomBC INFO:\n')
    #print(bottomBC)


    ## creates a vector with human readable dates
    dates = [datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") for t in time[:]]

    ## create a dataframe for boundary condition timeseries, this will simplify plotting
    topBC_DF = pd.DataFrame(np.column_stack([dates, topBC]), 
                                   columns=['Dates', 'topBC'])
    topBC_DF.topBC=topBC_DF.topBC.astype(float)
    
    topBC_DF=topBC_DF.set_index("Dates")
    
    
    bottomBC_DF = pd.DataFrame(np.column_stack([dates, bottomBC]), 
                                   columns=['Dates', 'bottomBC'])
    bottomBC_DF.bottomBC=bottomBC_DF.bottomBC.astype(float)
    
    bottomBC_DF=bottomBC_DF.set_index("Dates")
    
    return [ncfile,depths,dualDepths,time,psi,theta,iC,velocities,error,dates,topBC_DF,bottomBC_DF]



def showInitialCondition(iC,depths,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):

    plt.figure(figsize=(figureSizeHeigth,figureSizeWidth))

    plt.plot(iC,depths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize, color='b')
    plt.title('Initial condition',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['psi'].long_name + '  [' +ncfile.variables['psi'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.grid()
    plt.show()
    return


def showWaterSuction(timeIndex,psi,depths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):
    
    date = datetime.datetime.fromtimestamp(time[timeIndex])
    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))

    plt.plot(psi[timeIndex],depths[:], linewidth=lineWidth,linestyle=lineStyle, marker=markerType, markersize=markerSize, color='b')
    # convert time value in a human readable date to title the plot
    plt.title('Date: '+date.strftime('%Y-%m-%d %H:%M'),fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['psi'].long_name + '  [' +ncfile.variables['psi'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.grid()
    plt.show()
    return

def showWaterSuctionWithBCs(timeIndex,psi,depths,topBC_DF,bottomBC_DF,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):

    dateIndex = timeIndex
    date = datetime.datetime.fromtimestamp(time[timeIndex])
    plt.figure(figsize=(figureSizeWidth1,figureSizeHeigth1))
    
    axp = plt.subplot2grid((4, 6), (0, 0), rowspan=4, colspan=2)
    axp.plot(psi[timeIndex],depths[:],linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize, color='b')
    axp.set_xlabel(ncfile.variables['psi'].long_name + '  [' +ncfile.variables['psi'].units +']',fontsize=labelSize)
    axp.set_ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize)
    axp.set_title('Date: '+date.strftime('%Y-%m-%d %H:%M'),fontsize=titleSize)
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    axp.grid()
    
    axb = plt.subplot2grid((4, 6), (2,2), rowspan=2, colspan=4)
    axb.plot(bottomBC_DF[0:dateIndex+40], linewidth=lineWidth, label='_nolegend_')
    axb.set_xlabel("Time",fontsize=labelSize)
    axb.set_ylabel("[$m$]",fontsize=labelSize)
    axb.set_title('Bottom BC: water table',fontsize=titleSize)
    axb.vlines(x=dateIndex, ymin=np.min(bottomBC_DF['bottomBC'][:])*(0.5), ymax=(1.5), label=date.strftime('%Y-%m-%d %H:%M'), color='r',linewidth=lineWidth,)
    plt.legend()
    axb.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.setp(axb.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    
    axt = plt.subplot2grid((4, 6), (0, 2), rowspan=2,colspan=4,sharex=axb)
    axt.plot(topBC_DF[0:dateIndex+40],linewidth=lineWidth,)
    axt.set_xlabel("Time",fontsize=labelSize)
    axt.set_ylabel("[$mm$]",fontsize=labelSize)
    axt.set_title('Top BC: rainfall heigth',fontsize=titleSize)
    axt.vlines(x=dateIndex, ymin=np.min(topBC_DF['topBC'][:])*(0.5), ymax=(1.5), label=date.strftime('%Y-%m-%d %H:%M'), color='r',linewidth=lineWidth,)
    plt.legend()
    axt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.setp(axt.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    
    plt.tight_layout() 
    return

def showHydraulicHead(timeIndex,psi,depths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):

    date = datetime.datetime.fromtimestamp(time[timeIndex])
    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    
    plt.plot(psi[timeIndex]+depths[:]-depths[0],depths[:], linewidth=lineWidth,linestyle=lineStyle, marker=markerType, markersize=markerSize, color='b')
    # convert time value in a human readable date to title the plot
    plt.title('Date: '+date.strftime('%Y-%m-%d %H:%M'),fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel('Hydraulic head [m]',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.grid()
    plt.show()
    return

def showWaterContent(timeIndex,theta,depths,data,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):

    date = datetime.datetime.fromtimestamp(time[timeIndex])
    fig, ax=plt.subplots(figsize=(figureSizeWidth,figureSizeHeigth))
    figsize=(20,20)
    ax.plot(theta[timeIndex,0:depths[:].shape[0]-2],depths[0:depths[:].shape[0]-2], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize, color='r')
    waterLevel=theta[timeIndex,depths[:].shape[0]-1]
    ax.axhline(y=waterLevel, color='deepskyblue',linewidth=lineWidth,)
    
    # convert time value in a human readable date to title the plot
    plt.title('Date: '+date.strftime('%Y-%m-%d %H:%M'),fontsize=titleSize)
    # use variable attributes to label axis
    #plt.xlabel(ncfile.variables['water_heigth'].long_name + '  [' +ncfile.variables['water_heigth'].units +']',fontsize=labelSize )
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    
    plt.ylim(depths[0]-0.1,waterLevel+0.1)
    
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(-0.08,waterLevel, "{:.2f}".format(waterLevel), color="deepskyblue", transform=trans, 
        ha="right", va="center",fontsize=axisTicksSize)

    for i in range(1,np.size(data.index)-1):
        if data['Type'][i] == 'L':
            c = 'black'
            l = 'layer'
            plt.plot([np.min(theta[timeIndex,0:np.size(theta[timeIndex,])-1])-0.001,np.max(theta[timeIndex,])+0.001], [data['eta'][i],data['eta'][i]], color=c,linewidth=lineWidth-2)
            
    plt.legend(['$\\theta$', 'Total water depth','layer'], fontsize=legendSize,loc=4)
            
    plt.grid()
    plt.show()
    return

def showWaterContentWithBCs(timeIndex,theta,depths,bottomBC_DF,topBC_DF,data,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):
    waterLevel=theta[timeIndex,depths[:].shape[0]-1]

    date = datetime.datetime.fromtimestamp(time[timeIndex])
    dateIndex = timeIndex
    plt.figure(figsize=(figureSizeWidth1,figureSizeHeigth1))
    
    axp = plt.subplot2grid((4, 6), (0, 0), rowspan=4, colspan=2)
    axp.plot(theta[timeIndex,0:depths[:].shape[0]-2],depths[0:depths[:].shape[0]-2], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize, color='r')
    axp.axhline(y=theta[timeIndex,depths[:].shape[0]-1], color='deepskyblue',linewidth=lineWidth, linestyle='-')
    for i in range(1,np.size(data.index)-1):
        if data['Type'][i] == 'L':
            c = 'black'
            l = 'layer'
            axp.plot([np.min(theta[timeIndex,0:np.size(theta[timeIndex,])-1])-0.001,np.max(theta[timeIndex,])+0.001], [data['eta'][i],data['eta'][i]], color=c,linewidth=lineWidth-2)
        
    plt.legend(['$\\theta$', 'Total water depth','layer'], fontsize=legendSize,loc=3)
    axp.set_title('Date: '+date.strftime('%Y-%m-%d %H:%M'),fontsize=titleSize)
    plt.xlabel('$\\theta$ [$-$]',fontsize=labelSize )
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)

    # https://stackoverflow.com/questions/42877747/add-a-label-to-y-axis-to-show-the-value-of-y-for-a-horizontal-line-in-matplotlib
    trans = transforms.blended_transform_factory(
    axp.get_yticklabels()[0].get_transform(), axp.transData)
    axp.text(-0.08,waterLevel, "{:.2f}".format(waterLevel), color="deepskyblue", transform=trans, 
         ha="right", va="center",fontsize=axisTicksSize)
        
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    axp.grid()
    
    axb = plt.subplot2grid((4, 6), (2,2), rowspan=2, colspan=4)
    axb.plot(bottomBC_DF[0:dateIndex+40],linewidth=lineWidth, label='_nolegend_')
    axb.set_xlabel("Time",fontsize=labelSize)
    axb.set_ylabel("[$m$]",fontsize=labelSize)
    axb.set_title('Bottom BC: water table',fontsize=titleSize)
    axb.vlines(x=dateIndex, ymin=np.min(bottomBC_DF['bottomBC'][:])*(0.5), ymax=(1.5), label=date.strftime('%Y-%m-%d %H:%M'), color='r',linewidth=lineWidth)
    plt.legend()
    axb.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.setp(axb.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    
    axt = plt.subplot2grid((4, 6), (0, 2), rowspan=2,colspan=4,sharex=axb)
    axt.plot(topBC_DF[0:dateIndex+40],linewidth=lineWidth,)
    axt.set_xlabel("Time",fontsize=labelSize)
    axt.set_ylabel("[$mm$]",fontsize=labelSize)
    axt.set_title('Top BC: rainfall heigth',fontsize=titleSize)
    axt.vlines(x=dateIndex, ymin=np.min(topBC_DF['topBC'][:])*(0.5), ymax=(1.5), label=date.strftime('%Y-%m-%d %H:%M'), color='r',linewidth=lineWidth)
    plt.legend()
    axt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.setp(axt.get_xticklabels(), visible=False)
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    
    plt.tight_layout() 
    return

def showVelocities(timeIndex,velocities,dualDepths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):

    date = datetime.datetime.fromtimestamp(time[timeIndex])
    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))

    plt.plot(velocities[timeIndex],dualDepths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize, color='b')
    # convert time value in a human readable date to title the plot
    plt.title('Date: '+date.strftime('%Y-%m-%d %H:%M'),fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['velocities'].long_name + '  [' +ncfile.variables['velocities'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.grid()
    plt.show()
    return

def showError(error,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))

    plt.plot(time[:],abs(error[:]),'b',linewidth=lineWidth,)
    plt.semilogy()
    # convert time value in a human readable date to title the plot
    plt.title('Error over time',fontsize = titleSize)
    # use variable attributes to label axis
    #plt.xlabel(ncfile.variables['error'].long_name + '  [' +ncfile.variables['error'].units +']' )
    plt.ylabel(ncfile.variables['error'].long_name + '  [' +ncfile.variables['error'].units +']',fontsize = labelSize )
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    
    plt.grid()
    plt.show()
    return

    
def show(timeIndex,psi,theta,velocities,depths,dualDepths,data,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,figureSizeHeigth1,figureSizeWidth1):

    ## https://bokeh.pydata.org/en/latest/docs/user_guide/tools.html#built-in-tools
    date = datetime.datetime.fromtimestamp(time[timeIndex])
    hover = HoverTool(tooltips=[
            ("(x,y)", "($x, $y)"),
            ])

    p1 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots")
    
    p1.scatter(psi[timeIndex,:], depths[:], color="blue")
    p1.xaxis.axis_label = ncfile.variables['psi'].long_name + '  [' +ncfile.variables['psi'].units +']'
    p1.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p1.yaxis.axis_label = ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']'
    p1.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p1.title.text = 'Date: '+date.strftime('%Y-%m-%d %H:%M')
    p1.title.align = "center"
    p1.title.text_font_size = str(titleSize) + "px"
    tab1 = Panel(child=p1, title="Water suction")
    
    p2 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots")
    
    p2.scatter(psi[timeIndex,:]+depths[:]-depths[0], depths[:], color="blue")
    p2.xaxis.axis_label = 'Hydraulic head [m]'
    p2.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p2.yaxis.axis_label = ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']'
    p2.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p2.title.text = 'Date: '+date.strftime('%Y-%m-%d %H:%M')
    p2.title.align = "center"
    p2.title.text_font_size = str(titleSize) + "px"
    tab2 = Panel(child=p2, title="Hydraulic head")
    
    p3 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots")
    p3.scatter(theta[timeIndex,0:depths[:].shape[0]-2],depths[0:depths[:].shape[0]-2], color="red",legend='\u03B8 ')
    p3.line([theta[timeIndex,0:depths[:].shape[0]-2].min(),theta[timeIndex,0:depths[:].shape[0]-2].max()], [theta[timeIndex,depths[:].shape[0]-1],theta[timeIndex,depths[:].shape[0]-1]], color="deepskyblue",line_width=lineWidth, legend='Total water level')
    for i in range(1,np.size(data.index)-1):
        if data['Type'][i] == 'L':
            c = 'black'
            l = 'layer'
            p3.line([np.min(theta[timeIndex,0:np.size(theta[timeIndex,])-1])-0.001,np.max(theta[timeIndex,])+0.001], [data['eta'][i],data['eta'][i]], color=c,line_width=lineWidth-2,legend='Layer')

    p3.legend.location = "bottom_right"
    p3.legend.label_text_font_size = str(legendSize) + "px"
    p3.legend.click_policy="hide"
    #p3.xaxis.axis_label = '\u03B8 [-]'
    p3.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p3.yaxis.axis_label = ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']'
    p3.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p3.title.text = 'Date: '+date.strftime('%Y-%m-%d %H:%M')
    p3.title.align = "center"
    p3.title.text_font_size = str(titleSize) + "px"
    tab3 = Panel(child=p3, title="Water content and water depth")

    p4 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
           title="Mouse over the dots")
    
    p4.scatter(velocities[timeIndex], dualDepths[:], color="black")
    p4.xaxis.axis_label = ncfile.variables['velocities'].long_name + '  [' +ncfile.variables['velocities'].units +']'
    p4.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p4.yaxis.axis_label = ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']'
    p4.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p4.title.text = 'Date: '+date.strftime('%Y-%m-%d %H:%M')
    p4.title.align = "center"
    p4.title.text_font_size = str(titleSize) + "px"
    tab4 = Panel(child=p4, title="Darcy velocities")
    
    tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4 ])
    show(tabs)
    return