# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:42:31 2018

@author: Niccolo` Tubini, Concetta D'Amato and Riccardo Rigon
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
import matplotlib.dates as mdates
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
    
    dualDepths = ncfile.variables['dual_depth']
    
    time = ncfile.variables['time']
    
    psi = ncfile.variables['psi']

    theta = ncfile.variables['water_heigth']

    iC = ncfile.variables['psiIC']
    
    darcyVelocities = ncfile.variables['darcyVelocities']

    darcyVelocitiesCapillary = ncfile.variables['darcyVelocitiesCapillary']
    
    darcyVelocitiesGravity = ncfile.variables['darcyVelocitiesGravity']
    
    poreVelocities = ncfile.variables['poreVelocities']
    
    celerities = ncfile.variables['celerities']
    
    kinematicRatio = ncfile.variables['kinematicRatio']

    error = ncfile.variables['error']
    
    runOff = ncfile.variables['runOff']
    
    topBC = ncfile.variables['topBC']
    
    bottomBC = ncfile.variables['bottomBC']

    ## creates a vector with human readable dates
    datesHuman = [datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") for t in time[:]]

    ## creates a vector of dates
    dates = [pd.Timestamp(datetime.datetime.fromtimestamp(t)) for t in time]
    
    ## create a dataframe for boundary condition timeseries, this will simplify plotting
    topBC_DF = pd.DataFrame(np.column_stack([dates, topBC]), 
                                   columns=['Dates', 'topBC'])
    topBC_DF.topBC=topBC_DF.topBC.astype(float)
    
    topBC_DF=topBC_DF.set_index("Dates")
    
    
    bottomBC_DF = pd.DataFrame(np.column_stack([dates, bottomBC]), 
                                   columns=['Dates', 'bottomBC'])
    bottomBC_DF.bottomBC=bottomBC_DF.bottomBC.astype(float)
    
    bottomBC_DF=bottomBC_DF.set_index("Dates")
    
    return [ncfile,depths,dualDepths,time,psi,theta,iC,darcyVelocities,darcyVelocitiesCapillary,darcyVelocitiesGravity,celerities,kinematicRatio,error,runOff,dates,datesHuman,topBC_DF,bottomBC_DF]



def showInitialCondition(iC,depths,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeHeigth,figureSizeWidth))

    plt.plot(iC,depths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize, color='b')
    plt.plot(iC+depths[:]-depths[0],depths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize, color='g')
    plt.legend(['$\psi$', '$h$'],ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)
    plt.title('Initial condition',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['psi'].long_name + '  [' +ncfile.variables['psi'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.grid()
    plt.show()
    return


def showWaterSuction(timeIndex,date,psi,depths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    #color = ['b','g','r']
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])

    for i in range(0,len(timeIndex)):
        plt.plot(psi[timeIndex[i]],depths[:], linewidth=lineWidth,linestyle=lineStyle, marker=markerType, markersize=markerSize) 
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)
    plt.title('Water suction', fontsize=titleSize)
    
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['psi'].long_name + '  [' +ncfile.variables['psi'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.grid()
    plt.show()
    return


def showHydraulicHead(timeIndex,date,psi,depths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth,legendLoc):
   
    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    #color = ['b','g', 'r']
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])

    for i in range(0,len(timeIndex)):
        plt.plot(np.round(psi[timeIndex[i]]+depths[:]-depths[0],4),depths[:], linewidth=lineWidth,linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)
    plt.title('Hydraulic head', fontsize=titleSize)
    
    # use variable attributes to label axis
    plt.xlabel('Hydraulic head [m]',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.grid()
    plt.show()
    return

def showWaterContent(timeIndex,date,theta,depths,data,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    fig, ax=plt.subplots(figsize=(figureSizeWidth,figureSizeHeigth))
    figsize=(20,20)
    #color = ['b','g','r']
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])

    for i in range(0,len(timeIndex)):
        ax.plot(theta[timeIndex[i],0:depths[:].shape[0]-2],depths[0:depths[:].shape[0]-2], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)

    #for n in range(0,len(timeIndex)):  
    #    waterLevel=theta[timeIndex[n],depths[:].shape[0]-1]
    #    ax.axhline(y=waterLevel, linewidth=lineWidth)
    

    # convert time value in a human readable date to title the plot
    plt.title('Water content',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel('$\\theta$ [$-$]',fontsize=labelSize )
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.tick_params(axis='both', which='major', labelsize=axisTicksSize)
    
    #plt.ylim(depths[0]-0.1,waterLevel+0.1)
    #trans = transforms.blended_transform_factory(
    #   ax.get_yticklabels()[0].get_transform(), ax.transData)
    #ax.text(-0.08,waterLevel, "{:.2f}".format(waterLevel), color="deepskyblue", transform=trans, 
    #   ha="right", va="center",fontsize=axisTicksSize)
    for n in range(0,len(timeIndex)):
        for i in range(1,np.size(data.index)-1):
            if data['Type'][i] == 'L':
                c = 'black'
                l = 'layer'
                plt.plot([np.min(theta[timeIndex[n],0:np.size(theta[timeIndex[n],])-1])-0.001,np.max(theta[timeIndex[n],])+0.001], [data['eta'][i],data['eta'][i]], color=c,linewidth=lineWidth-2)
         
    plt.grid()
    plt.show()
    return


def showDarcyVelocities(timeIndex,date,velocities,dualDepths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])
    for i in range(0,len(timeIndex)):
        plt.plot(velocities[timeIndex[i]],dualDepths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)

    plt.title('Darcy flux',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['darcyVelocities'].long_name + '  [' +ncfile.variables['darcyVelocities'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.legend(date, loc=3)
    plt.grid()
    plt.show()
    return

def showCapillaryVelocities(timeIndex,date,velocities,dualDepths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])
    for i in range(0,len(timeIndex)):
        plt.plot(velocities[timeIndex[i]],dualDepths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)

    plt.title('Fluxes due to capillary gradient',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['darcyVelocitiesCapillary'].long_name + '  [' +ncfile.variables['darcyVelocitiesCapillary'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.legend(date, loc=3)
    plt.grid()
    plt.show()
    return

def showGravityVelocities(timeIndex,date,velocities,dualDepths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])
    for i in range(0,len(timeIndex)):
        plt.plot(velocities[timeIndex[i]],dualDepths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)

    plt.title('Fluxes due to gravity gradient',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['darcyVelocitiesGravity'].long_name + '  [' +ncfile.variables['darcyVelocitiesGravity'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.legend(date, loc=3)
    plt.grid()
    plt.show()
    return

def showPoreVelocities(timeIndex,date,velocities,dualDepths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])
    for i in range(0,len(timeIndex)):
        plt.plot(velocities[timeIndex[i]],dualDepths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)

    plt.title('Fluxes due to gravity gradient',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['poreVelocities'].long_name + '  [' +ncfile.variables['poreVelocities'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.legend(date, loc=3)
    plt.grid()
    plt.show()
    return

def showCelerities(timeIndex,date,velocities,dualDepths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])
    for i in range(0,len(timeIndex)):
        plt.plot(velocities[timeIndex[i]],dualDepths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)

    plt.title('Celerity',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['celerities'].long_name + '  [' +ncfile.variables['celerities'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.legend(date, loc=3)
    plt.grid()
    plt.show()
    return

def showKinematicRatio(timeIndex,date,kinematicRatio, dualDepths,time,ncfile,labelSize,titleSize,legendSize,axisTicksSize,lineWidth,lineStyle,markerSize,markerType,figureSizeHeigth,figureSizeWidth, legendLoc):

    plt.figure(figsize=(figureSizeWidth,figureSizeHeigth))
    colormap = plt.cm.nipy_spectral
    plt.gca().set_prop_cycle("color",[colormap(i) for i in np.linspace(0, 0.9, len(timeIndex))])
    for i in range(0,len(timeIndex)):
        plt.plot(kinematicRatio[timeIndex[i]],dualDepths[:], linewidth=lineWidth, linestyle=lineStyle, marker=markerType, markersize=markerSize)
    plt.legend(date,ncol=1, loc=legendLoc, 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=legendSize)

    plt.title('Kinematic ratio',fontsize=titleSize)
    # use variable attributes to label axis
    plt.xlabel(ncfile.variables['kinematicRatio'].long_name + '  [' +ncfile.variables['kinematicRatio'].units +']',fontsize=labelSize)
    plt.ylabel(ncfile.variables['depth'].long_name + '  [' +ncfile.variables['depth'].units +']',fontsize=labelSize )
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(fontsize=axisTicksSize)
    plt.yticks(fontsize=axisTicksSize)
    plt.legend(date, loc=3)
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