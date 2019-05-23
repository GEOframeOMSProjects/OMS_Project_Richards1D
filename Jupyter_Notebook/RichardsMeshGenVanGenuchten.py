# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:54:52 2018

This is used to create Richards 1D mesh using Van Genuchten SWRC

@author: Niccolo` Tubini and Riccardo Rigon
@license: creative commons 4.0
"""
import pandas as pd
import numpy as np

from netCDF4 import Dataset

from bokeh.io import output_notebook,output_file, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models.widgets import Panel, Tabs
#from bokeh.io import output_file, show
#from bokeh.plotting import figure
from bokeh.models import BoxSelectTool
from bokeh.models import HoverTool

output_notebook()

def buildData(data):
    '''
    This function creates the geometry of 1D grid for a finite volume numerical 
    scheme.
    
    
    data is a pandas dataframe
    
    
    
    return:
    
    eta: vertical coordinate of control volumes centroids positive upward with 
        origin set at soil surface.
    
    etaDual: vertical coordinate of control volumes interfaces positive upward with 
        origin set at soil surface.
        
    length: control volume lenght.

    spaceDelta: is the distance between two adjacent control volumes. 
                This quantity is used to compute gradients.    
    
    z: vertical coordinate of control volumes centroids positive upward with 
        origin set at soil column bottom. This is the spatial coordinate used to
        write Richards' equation
    
    zDual: vertical coordinate of control volumes interfaces positive upward with 
        origin set at soil column bottom.  
        
    '''
    
    ## list containing centroids coordinates
    eta = [] 
    ## list containing control volumes interface coordinates
    etaDual = []

    
    for i in range(np.size(data.index)-1,0,-1):
        if data['Type'][i]=='L' and data['Type'][i-1]=='L':
            #print("Layer layer")
            deta = ( data['eta'][i]-data['eta'][i-1])/data['N'][i-1]
            eta=np.append(eta, np.linspace(data['eta'][i]-deta/2,data['eta'][i-1]+deta/2,num=data['N'][i-1],endpoint=True) )
            etaDual=np.append(etaDual, np.linspace(data['eta'][i],data['eta'][i-1],num=data['N'][i-1]+1,endpoint=True) )
        elif data['Type'][i]=='L' and data['Type'][i-1]=='M':
            #print("Layer Meas")
            deta = ( data['eta'][i]-data['eta'][i-1])/data['N'][i-1]
            eta=np.append(eta, np.linspace(data['eta'][i]-deta/2,data['eta'][i-1],num=data['N'][i-1],endpoint=True) )
            etaDual=np.append(etaDual, np.linspace(data['eta'][i],data['eta'][i-1]+deta/2,num=data['N'][i-1],endpoint=True) )
        elif data['Type'][i]=='M' and data['Type'][i-1]=='L':
            #print("Meas layer")
            deta = ( data['eta'][i]-data['eta'][i-1])/data['N'][i-1]
            eta=np.append(eta, np.linspace(data['eta'][i],data['eta'][i-1]+deta/2,num=data['N'][i-1],endpoint=True) )
            etaDual=np.append(etaDual, np.linspace(data['eta'][i]-deta/2,data['eta'][i-1],num=data['N'][i-1],endpoint=True) )
        else:
            print("ERROR!!")  
        
    ## to eliminate doubles
    eta=[ii for n,ii in enumerate(eta) if ii not in eta[:n]]
    etaDual=[ii for n,ii in enumerate(etaDual) if ii not in etaDual[:n]]


    ## control volume length
    length = []

    for i in range(0,np.size(etaDual)-1):
        length = np.append(length, np.abs(etaDual[i]-etaDual[i+1]) )

    
    ## space length: is used to cumpute gradients
    spaceDelta = []

    for i in range(0,np.size(etaDual)):
        if i==0:
            spaceDelta = np.append(spaceDelta, np.abs(etaDual[i]-eta[i]) )
        elif i==np.size(etaDual)-1:
            spaceDelta = np.append(spaceDelta, np.abs(etaDual[i]-eta[i-1]) )
        else:
             spaceDelta = np.append(spaceDelta, np.abs(eta[i-1]-eta[i]) )
    eta= np.append(eta,data['eta'][0])
    z = []
    zDual = []
    for i in range(0, np.size(eta)):
        z = np.append(z,eta[i]-data['eta'][np.size(data['eta'])-1])
    
    for i in range(0, np.size(etaDual)):
        zDual = np.append(zDual,etaDual[i]-data['eta'][np.size(data['eta'])-1])

    return [eta,etaDual,length,spaceDelta,z,zDual]


def showMesh(data,z,eta,etaDual,labelSize,titleSize,legendSize,axisTicksSize,lineWidth):
    x=np.zeros(np.size(z))

    hover = HoverTool(tooltips=[
        ("(x,y)", "($x, $y)"),
    ])

    p1 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
           title="Mouse over the dots",x_range=(-0.7, 0.7))
    p1.scatter(x,eta, color="blue", marker='circle',legend='Centroids')
    p1.scatter(x,etaDual, color="fuchsia", marker='cross',legend='CV interface')
    for i in range(0,np.size(data.index)):
        if data['Type'][i] == 'L':
            c = 'red'
            l = 'layer'
        elif data['Type'][i] == 'M':
            c = 'green'
            l = 'meas. point'
        
        p1.line([-0.2,0.2], [data['eta'][i],data['eta'][i]], color=c,line_width=lineWidth,legend=l)
    p1.line([-0.2,0.2], [data['eta'][data['eta'].size-1],data['eta'][data['eta'].size-1]], color="red",line_width=lineWidth,legend='layer')

    p1.yaxis.axis_label = '\u03b7 [m]'
    p1.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p1.title.text = 'Grid geometry'
    p1.title.align = "center"
    p1.title.text_font_size = str(titleSize) + "px"

    p1.legend.location = "top_left"
    p1.legend.label_text_font_size = str(legendSize) + "px"
    p1.legend.click_policy="hide"
    show(p1)
         


## This at the ends just gived the coordinate of the measured point and the value of psi measured
def initialPsi(data):
    #psiIC = []
    coordMeasPoint = []
    psiMeas = []

    for i in data.index:
        if data['Type'][i] == 'M':
            coordMeasPoint.append(data['eta'][i])
            psiMeas.append(data['psi'][i])
        
    coordMeasPoint = np.append(coordMeasPoint, data['eta'][np.size(data['eta'])-1])
    psiMeas = np.append(psiMeas,data['psi'][np.size(data['psi'])-1])
    return [coordMeasPoint,psiMeas]


## Initial condition idrostatic
def setInitialCondition(data,eta,z,icType):
    [coordMeasPoint,psiMeas] = initialPsi(data)
    ## hydrostatic
    if icType=='hydrostatic':
        psiIC = []
        for i in range(0,np.size(eta)-1):
            psiIC = np.append(psiIC,data['psi'][np.size(data['psi'])-1]+(data['eta'][np.size(data['eta'])-1]-eta[i]))

        psiIC = np.append(psiIC,data['psi'][0] )
        
    ##constant
    elif icType=='constant':
        psiIC = []
        for i in range(0,np.size(eta)-1):
            psiIC = np.append(psiIC,data['psi'][np.size(data['psi'])-1])

        psiIC = np.append(psiIC,data['psi'][0] )
    
    ## linear interpolation
    elif icType=='linear interpolation':
        psiIC = []
        for i in range(np.size(coordMeasPoint)-1,-1,-1):
            etaInterp = []
            if i==np.size(coordMeasPoint)-1:
                etaInterp = eta[0:eta.tolist().index(coordMeasPoint[i-1])]
                etap = [coordMeasPoint[i],coordMeasPoint[i-1]]
                fp = [psiMeas[i],psiMeas[i-1]]
                psiIC = np.append(psiIC, np.interp(etaInterp,etap,fp) )
                #for j in range(0,np.size(etaInterp)):
                #    psiIC = np.append(psiIC,psiMeas[i]+(coordMeasPoint[i]-etaInterp[j]))
            elif i== 0:
                etaInterp = eta[eta.tolist().index(coordMeasPoint[i]):np.size(eta)-1]
                etap = [coordMeasPoint[i],0]
                #fp = [psiMeas[i],data['psi'][0]]
                fp = [psiMeas[i],-z[np.size(z)-1]]
                psiIC = np.append(psiIC, np.interp(etaInterp,etap,fp) )
            else:
                etaInterp = eta[eta.tolist().index(coordMeasPoint[i-1]):eta.index(coordMeasPoint[i])]
                etap = [coordMeasPoint[i],coordMeasPoint[i-1]]
                fp = [psiMeas[i],psiMeas[i-1]]
                psiIC = np.append(psiIC, np.interp(etaInterp,etap,fp) )

        psiIC = np.append(psiIC,data['psi'][0] )
    
    else:
        print('icType is not hydrostatic or constant or linear interpolation')
    
    return psiIC

## plot initial condition
def showInitialCondition(psiIC,eta,icType,labelSize,titleSize,legendSize,axisTicksSize,lineWidth):
    hover = HoverTool(tooltips=[
        ("(x,y)", "($x, $y)"),
    ])

    p1 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
           title="Mouse over the dots")
    p1.scatter(psiIC,eta, color="blue")

    p1.xaxis.axis_label = '\u03C8 [m]'
    p1.xaxis.axis_label_text_font_size = str(labelSize) + "px"

    p1.yaxis.axis_label = '\u03b7 [m]'
    p1.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p1.title.text = 'Initial condition for \u03C8: '+icType
    p1.title.align = "center"
    p1.title.text_font_size = str(titleSize) + "px"
    show(p1)
    return    
    
## plot total head 
def showTotalHead(psiIC,z,eta,icType,labelSize,titleSize,legendSize,axisTicksSize,lineWidth):
    hover = HoverTool(tooltips=[
        ("(x,y)", "($x, $y)"),
    ])

    p1 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
           title="Mouse over the dots")
    p1.scatter(psiIC+z,eta, color="blue")

    p1.xaxis.axis_label = 'h [m]'
    p1.xaxis.axis_label_text_font_size = str(labelSize) + "px"

    p1.yaxis.axis_label = '\u03b7  [m]'
    p1.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p1.title.text = 'Total hydraulic head:'+icType
    p1.title.align = "center"
    p1.title.text_font_size = str(titleSize) + "px"


    show(p1)
    return

## set parameters
def setParameters(data,eta):
    thetaS = []
    thetaR = []
    Ks = []
    alphaSS = []
    betaSS = []
    par1SWRC = []
    par2SWRC = []
    par3SWRC = []
    par4SWRC = []
    par5SWRC = []
    par6SWRC = []
    par7SWRC = []
    par8SWRC = []


    et = []


    coordLayer = []
    tempThetaS = []
    tempThetaR = []
    tempKs = []
    tempAlphaSS = []
    tempBetaSS = []
    tempPar1 = []
    tempPar2 = []
    tempPar3 = []
    tempPar4 = []
    tempPar5 = []
    tempPar6 = []
    tempPar7 = []
    tempPar8 = []
    tempEt = []

    for i in data.index:
        if data['Type'][i] == 'L':
            coordLayer.append(data['eta'][i])
            tempThetaS.append(data['thetaS'][i])
            tempThetaR.append(data['thetaR'][i])
            tempKs.append(data['Ks'][i])
            tempAlphaSS.append(data['alphaSpecificStorage'][i])
            tempBetaSS.append(data['betaSpecificStorage'][i])
            tempPar1.append(data['n'][i])
            tempPar2.append(data['alpha'][i])
            tempPar3.append(-999.0)
            tempPar4.append(-999.0)
            tempPar5.append(-999.0)
            tempPar6.append( -1/data['alpha'][i] * ( (data['n'][i]-1)/data['n'][i] )**(1/data['n'][i]) )
            tempPar7.append(-999.0)
            tempPar8.append(-999.0)
            tempEt.append(data['et'][i])
            
     
    for i in range(np.size(coordLayer)-1,0,-1):
        for j in range(0,np.size(eta)):
            if(eta[j]>coordLayer[i] and eta[j]<coordLayer[i-1] ):
                thetaS.append(tempThetaS[i-1])
                thetaR.append(tempThetaR[i-1])
                Ks.append(tempKs[i-1])
                alphaSS.append(tempAlphaSS[i-1])
                betaSS.append(tempBetaSS[i-1])
                par1SWRC.append(tempPar1[i-1])
                par2SWRC.append(tempPar2[i-1])
                par3SWRC.append(tempPar3[i-1])
                par4SWRC.append(tempPar4[i-1])
                par5SWRC.append(tempPar5[i-1])
                par6SWRC.append(tempPar6[i-1])
                par7SWRC.append(tempPar7[i-1])
                par8SWRC.append(tempPar8[i-1])
                et.append(tempEt[i-1])
    
        ## add et coeff for free water
    et = np.append(et,0)   
        
    return [thetaS, thetaR, Ks, alphaSS, betaSS, par1SWRC, par2SWRC, par3SWRC, par4SWRC, par5SWRC,  par6SWRC,  par7SWRC,  par8SWRC,  et]

'''
plot parameters 
This function allows to plot the distribution of SWRC parameters in the soil column.

These plots are drawn with bokeh library https://bokeh.pydata.org/en/latest/
and they are interactive.
'''
def showParameters(thetaS,thetaR,Ks,alphaSpecificStorage,betaSpecificStorage, n,alpha, psiStar,et,eta,labelSize,titleSize,axisTicksSize,lineWidth):
    hover = HoverTool(tooltips=[
    ("(x,y)", "($x, $y)"),
    ])

    p1 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
           title="Mouse over the dots")
    
    p1.scatter(thetaS, eta[0:np.size(eta)-1], color="blue", line_width=lineWidth)
    p1.xaxis.axis_label = ' water content at saturation [-]'
    p1.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p1.yaxis.axis_label = '\u03b7 [m]'
    p1.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p1.title.text = 'Water content at saturation'
    p1.title.align = "center"
    p1.title.text_font_size = str(titleSize) + "px"
    tab1 = Panel(child=p1, title="Theta_s")

    p2 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots")
    p2.scatter(thetaR, eta[0:np.size(eta)-1],line_width=lineWidth, color="red")
    p2.xaxis.axis_label = 'residual water content [-]'
    p2.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p2.yaxis.axis_label = '\u03b7 [m]'
    p2.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p2.title.text = 'Residual water content'
    p2.title.align = "center"
    p2.title.text_font_size = str(titleSize) + "px"
    tab2 = Panel(child=p2, title="Theta_r")

    p3 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots" )
    p3.scatter(Ks, eta[0:np.size(eta)-1],line_width=lineWidth, color="red")
    p3.xaxis.axis_label = 'Ks [m/s]'
    p3.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p3.yaxis.axis_label = '\u03b7 [m]'
    p3.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p3.title.text = 'Saturated hydraulic conductivity'
    p3.title.align = "center"
    p3.title.text_font_size = str(titleSize) + "px"
    tab3= Panel(child=p3, title="Ks")

    p4 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots" )
    p4.scatter(n, eta[0:np.size(eta)-1],line_width=lineWidth, color="red")
    p4.xaxis.axis_label = 'n [-] '
    p4.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p4.yaxis.axis_label = '\u03b7 [m]'
    p4.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p4.title.text = 'Van Genuchten n'
    p4.title.align = "center"
    p4.title.text_font_size = str(titleSize) + "px"
    tab4= Panel(child=p4, title="n")

    p5 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots" )
    p5.scatter(alpha, eta[0:np.size(eta)-1],line_width=lineWidth, color="red")
    p5.xaxis.axis_label = '\u03B1 [m] '
    p5.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p5.yaxis.axis_label = '\u03b7 [m]'
    p5.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p5.title.text = 'Van Genuchten \u03B1'
    p5.title.align = "center"
    p5.title.text_font_size = str(titleSize) + "px"
    tab5= Panel(child=p5, title="\u03B1")
    
    p6 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots" )
    p6.scatter(psiStar, eta[0:np.size(eta)-1],line_width=lineWidth, color="red")
    p6.xaxis.axis_label = '\u03C8* [m] '
    p6.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p6.yaxis.axis_label = '\u03b7 [m]'
    p6.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p6.title.text = 'Van Genuchten \u03C8*'
    p6.title.align = "center"
    p6.title.text_font_size = str(titleSize) + "px"
    tab6= Panel(child=p6, title="\u03C8*")

    p7 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots" )
    p7.scatter(alphaSpecificStorage, eta[0:np.size(eta)-1],line_width=lineWidth, color="red")
    p7.xaxis.axis_label = '\u03b1SS [1/Pa]'
    p7.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p7.yaxis.axis_label = '\u03b7 [m]'
    p7.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p7.title.text = 'Compressibility of soil'
    p7.title.align = "center"
    p7.title.text_font_size = str(titleSize) + "px"
    tab7= Panel(child=p7, title="\u03b1SpecStor")

    p8 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots" )
    p8.scatter(betaSpecificStorage, eta[0:np.size(eta)-1],line_width=lineWidth, color="red")
    p8.xaxis.axis_label = '\u03b2SS [1/Pa]'
    p8.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p8.yaxis.axis_label = '\u03b7 [m]'
    p8.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p8.title.text = 'Compressibility of water'
    p8.title.align = "center"
    p8.title.text_font_size = str(titleSize) + "px"
    tab8= Panel(child=p8, title="\u03b2SpecStor")

    p9 = figure(plot_width=600, plot_height=600,tools=['pan,wheel_zoom,box_zoom,reset',hover],
            title="Mouse over the dots" )
    p9.scatter(et, eta,line_width=lineWidth, color="red")
    p9.xaxis.axis_label = 'et coeff. [1/s]'
    p9.xaxis.axis_label_text_font_size = str(labelSize) + "px"
    p9.yaxis.axis_label = '\u03b7 [m]'
    p9.yaxis.axis_label_text_font_size = str(labelSize) + "px"
    p9.title.text = 'Source sink term'
    p9.title.align = "center"
    p9.title.text_font_size = str(titleSize) + "px"
    tab9= Panel(child=p9, title="et")
    
    tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 ])
    show(tabs)
    
'''
Save all grid data in a NetCDF file
'''
def writeGridNetCDF(eta,etaDual,z,zDual,spaceDelta,deltaZ,psiIC,thetaS,thetaR,Ks,alphaSpecificStorage,betaSpecificStorage,n,alpha,par3SWRC,par4SWRC,par5SWRC,psiStar,par7SWRC,par8SWRC,et,outputFileName,outputTitle,outputInstitution,outputSummary,folderPath,outputDate,inputFileName):
        # the output array to write will be nx x ny
    dim = np.size(eta);
    dim1 = np.size(thetaS)
    # open a new netCDF file for writing.
    ncfile = Dataset(outputFileName,'w') 
    
    # Create global attributes
    ncfile.title = outputTitle + '\\n' + 'input file' + folderPath + '\\' + inputFileName
    ncfile.institution =  outputInstitution
    ncfile.summary = outputSummary
    #ncfile.acknowledgment = ""
    ncfile.date_created = outputDate

    
    # create the z dimensions.
    ncfile.createDimension('z',dim)
    ncfile.createDimension('zz',dim1)
    
    # create the variable
    # first argument is name of variable, second is datatype, third is
    # a tuple with the names of dimensions.
    dataEta = ncfile.createVariable('eta','f8',('z'))
    dataEta.unit = 'm'
    dataEta.long_name = '\u03b7 coordinate of volume centroids: zero is at soil surface and and positive upward'
    
    dataEtaDual = ncfile.createVariable('etaDual','f8',('z'))
    dataEtaDual.unit = 'm'
    dataEtaDual.long_name = '\u03b7 coordinate of volume interfaces: zero is at soil surface and and positive upward. '
    
    dataZ = ncfile.createVariable('z','f8',('z'))
    dataZ.unit = 'm'
    dataZ.long_name = 'z coordinate  of volume centroids: zero is at the bottom of the column and and positive upward'
    
    dataZDual = ncfile.createVariable('zDual','f8',('z'))
    dataZDual.unit = 'm'
    dataZDual.long_name = 'z coordinate of volume interfaces: zero is at soil surface and and positive upward.'
    
    
    dataPsiIC = ncfile.createVariable('psiIC','f8',('z'))
    dataPsiIC.units = 'm'
    dataPsiIC.long_name = 'initial condition for water suction'
    
    dataSpaceDelta = ncfile.createVariable('spaceDelta','f8',('z'))
    dataSpaceDelta.unit = 'm'
    dataSpaceDelta.long_name = 'Distance between consecutive controids, is used to compute gradients'
    
    dataEt = ncfile.createVariable('et','f8',('z'))
    dataEt.units = '1/s'
    dataEt.long_name = 'Coefficient to simulate ET'
    
    
    
    
    dataDeltaZ = ncfile.createVariable('deltaZ','f8',('zz'))
    dataDeltaZ.unit = 'm'
    dataDeltaZ.long_name = 'Length of each control volume'
    
    dataThetaS = ncfile.createVariable('thetaS','f8',('zz'))
    dataThetaS.units = '-'
    dataThetaS.long_name = 'adimensional water content at saturation'
    
    dataThetaR = ncfile.createVariable('thetaR','f8',('zz'))
    dataThetaR.units = '-'
    dataThetaR.long_name = 'adimensional residual water content'
    
    dataKs = ncfile.createVariable('Ks','f8',('zz'))
    dataKs.units = 'm/s'
    dataKs.long_name = 'hydraulic conductivity at saturation'
    
    dataAlphaSS = ncfile.createVariable('alphaSpecificStorage','f8',('zz'))
    dataAlphaSS.units = '1/Pa'
    dataAlphaSS.long_name = 'Aquitard compressibility'
    
    dataBetaSS = ncfile.createVariable('betaSpecificStorage','f8',('zz'))
    dataBetaSS.units = '1/Pa'
    dataBetaSS.long_name = 'Water compressibility'
    
    ## enter the long_name of par1SWRC, as an example  Parameter n of Van Genuchten model
    dataPar1SWRC = ncfile.createVariable('par1SWRC','f8',('zz'))
    dataPar1SWRC.units = '-'
    dataPar1SWRC.long_name = 'Parameter n of Van Genuchten model'
    
    ## enter the long_name of par2SWRC, as an example  Parameter alpha of Van Genuchten model
    dataPar2SWRC = ncfile.createVariable('par2SWRC','f8',('zz'))
    dataPar2SWRC.units = 'm'
    dataPar2SWRC.long_name = 'Parameter alpha of Van Genuchten model'
    
    ## enter the long_name of par3SWRC, for Van Genuchten is nan
    dataPar3SWRC = ncfile.createVariable('par3SWRC','f8',('zz'))
    dataPar3SWRC.units = '-'
    dataPar3SWRC.long_name = 'no value'
    
    ## enter the long_name of par4SWRC, for Van Genuchten is nan
    dataPar4SWRC = ncfile.createVariable('par4SWRC','f8',('zz'))
    dataPar4SWRC.units = '-'
    dataPar4SWRC.long_name = 'no value'
    
    ## enter the long_name of par5SWRC, for Van Genuchten is nan
    dataPar5SWRC = ncfile.createVariable('par5SWRC','f8',('zz'))
    dataPar5SWRC.units = '-'
    dataPar5SWRC.long_name = 'no value'
    
    ## enter the long_name of par6SWRC, critical value of psi for Van Genuchten model
    dataPar6SWRC = ncfile.createVariable('par6SWRC','f8',('zz'))
    dataPar6SWRC.units = 'm'
    dataPar6SWRC.long_name = 'Critical value of psi, where moisture capacity is null'
    
    ## enter the long_name of par7SWRC, for Van Genuchten is nan
    dataPar7SWRC = ncfile.createVariable('par7SWRC','f8',('zz'))
    dataPar7SWRC.units = '-'
    dataPar7SWRC.long_name = 'no value'
    
    ## enter the long_name of par8SWRC, for Van Genuchten is nan
    dataPar8SWRC = ncfile.createVariable('par8SWRC','f8',('zz'))
    dataPar8SWRC.units = '-'
    dataPar8SWRC.long_name = 'no value'




    ## write data to variable.
    for i in range(0,dim):
        dataEta[i] = eta[i]
        dataEtaDual[i] = etaDual[i]
        dataZ[i] = z[i]
        dataZDual[i] = zDual[i]
        dataPsiIC[i] = psiIC[i]
        dataSpaceDelta[i] = spaceDelta[i]
        dataEt[i] = et[i]
   

    for i in range(0,dim1):
        dataDeltaZ[i] = deltaZ[i]
        dataThetaS[i] = thetaS[i]
        dataThetaR[i] = thetaR[i]
        dataKs[i] = Ks[i]
        dataAlphaSS[i] = alphaSpecificStorage[i]
        dataBetaSS[i] = betaSpecificStorage[i]
        dataPar1SWRC[i] = n[i]
        dataPar2SWRC[i] = alpha[i]
        dataPar3SWRC[i] = par3SWRC[i]
        dataPar4SWRC[i] = par4SWRC[i]
        dataPar5SWRC[i] = par5SWRC[i]
        dataPar6SWRC[i] = psiStar[i]
        dataPar7SWRC[i] = par7SWRC[i]
        dataPar8SWRC[i] = par8SWRC[i]
   
        
    ## close the file.
    ncfile.close()
    print ('*** SUCCESS writing!  '+outputFileName)
    return