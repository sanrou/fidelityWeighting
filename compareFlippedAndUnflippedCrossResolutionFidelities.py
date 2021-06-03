# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:13:28 2021
Compare flipped and non-flipped (weights signed or not) cross-resolution exponent analysis results.
Load fidelity arrays made with crossParcellationResolutionAnalysisExponents.py.

@author: rouhinen
"""

import numpy as np
import matplotlib.pyplot as plt

## Settings used in creating the plvArrays.
resolutions = ['100', '200', '400', '597', '775', '942']
exponents = [0, 1, 2, 4, 8, 16, 32]


""" Load PLV arrays from files. """
plvArray_meth1_u = np.load("C:\\temp\\fWeighting\\plvArrays\\exponents0to32Unsigned\\plvArray_meth1.npy")
plvArray_meth2_u = np.load("C:\\temp\\fWeighting\\plvArrays\\exponents0to32Unsigned\\plvArray_meth2m.npy")
plvArray_meth1_f = np.load("C:\\temp\\fWeighting\\plvArrays\\exponents0to32Signed\\plvArray_meth1.npy")
plvArray_meth2_f = np.load("C:\\temp\\fWeighting\\plvArrays\\exponents0to32Signed\\plvArray_meth2m.npy")

""" Get relative fidelity means differences of flipped and unflipped. """
def nonZeroMeans(zeroBufferedData, exponents, resolutions):
  """ zeroBufferedData shape: exponents x subjects x resolutions x resolutions x maxResolution. """
  means = np.zeros((len(exponents), len(resolutions),len(resolutions)))
  for i1, resolution1 in enumerate(resolutions):
      for i2, resolution2 in enumerate(resolutions):
        for iexp, exponent in enumerate(exponents):
          nonZero = zeroBufferedData[iexp,:,i1,i2,:]
          nonZero = nonZero[nonZero != 0]
          means[iexp,i1,i2] = round(np.mean(nonZero), 4)
  return means

means_meth1_u = nonZeroMeans(plvArray_meth1_u, exponents, resolutions)    # Method 1 modeled resolution, non-flipped
means_meth2_u = nonZeroMeans(plvArray_meth2_u, exponents, resolutions)    # Method 2 modeled resolution, non-flipped
means_meth1_f = nonZeroMeans(plvArray_meth1_f, exponents, resolutions)    # Method 1 modeled resolution, flipped
means_meth2_f = nonZeroMeans(plvArray_meth2_f, exponents, resolutions)    # Method 2 modeled resolution, flipped

means_meth1_r = means_meth1_u / means_meth1_f
means_meth2_r = means_meth2_u / means_meth2_f


""" Plot """
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
params = {'legend.fontsize':'7',
         'figure.figsize':(2.0*len(exponents), 2.2),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'font.family':'Arial',
         'pdf.fonttype':42,
         'ps.fonttype':42}
pylab.rcParams.update(params)

def heat_plot_exp(data, tickLabels, titleStrings, vmin=0.1, vmax=0.6, decimals=2):
    # Data 3D, with first dimension sub-plots.
    columns = len(data)
    
    # Set a threshold where text should be black instead of white. 
    middle = (vmax+vmin)/2
    textToKT = (vmax-vmin) * 0.15
    
    fig, ax = plt.subplots(1, columns)
    for i, datum in enumerate(data):
        ax[i].imshow(datum[::-1,:], cmap='seismic', vmin=vmin, vmax=vmax)  # Visualize Y-axis down to up.
        
        # Show all ticks...
        ax[i].set_xticks(np.arange(len(tickLabels)))
        ax[i].set_yticks(np.arange(len(tickLabels)))
        # ... and label them with the respective list entries
        ax[i].set_xticklabels(tickLabels)
        ax[i].set_yticklabels(tickLabels[::-1])    # Reverse y-axis labels.
        
        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax[i].get_xticklabels(), rotation=0, ha="right",
        #          rotation_mode="anchor")
        
        # Loop over datum dimensions and create text annotations.
        for ii in range(len(tickLabels)):
            for j in range(len(tickLabels)):
                value = round(datum[-ii-1, j], decimals)
                tcolor = "w" if np.abs(value-middle) > textToKT else "k"    # Set text color to white if not near middle threshold, else to black.
                ax[i].text(j, ii, value, ha="center", va="center", color=tcolor, fontsize=7)
        
        ax[i].set_title(titleStrings[i])
        ax[i].set_xlabel('Modeling resolution')
        ax[i].set_ylabel('Simulation resolution')
    
    fig.tight_layout()
    plt.show()

# Method 1
maxDiffFromOne = np.max(np.abs([means_meth1_r, means_meth2_r])) -1
vminR = 1-maxDiffFromOne
vmaxR = 1+maxDiffFromOne

meth1Strings = ['Unflipped/Flipped fidelity,\n method1, exponent ' + str(exponent) for exponent in exponents]
heat_plot_exp(means_meth1_r, resolutions, meth1Strings, vmin=vminR, vmax=vmaxR)

# Method 2
meth2Strings = ['Unflipped/Flipped fidelity,\n method2, exponent ' + str(exponent) for exponent in exponents]
heat_plot_exp(means_meth2_r, resolutions, meth2Strings, vmin=vminR, vmax=vmaxR)



## Means of relative all values, diagonal, upper and lower triangles without diagonal. 
def means_withTriangles(data, resolutions, exponents, verbose=True):
  """ data : exponents x resolutions x resolutions. 
      Output : 4 x resolutions, with first dimension means of whole array, upper triangle without 
      diagonal indices, diagonal, and lower triangle without diagonal indices. 
      Upper diagonal: higher modeling resolution than simulation resolution. """
  decimals = 3
  mean_byExp = np.round([np.mean(means) for means in data], decimals)
  
  if verbose==True:
    for i, mean in enumerate(mean_byExp):
      print(f'Mean fidelity with exponent {exponents[i]} whole {mean}')
  
  # Upper, lower, diagonal means.
  iup = np.triu_indices(len(resolutions), 1)  # Upper triangle without diagonal indices.
  idi = np.diag_indices(len(resolutions))  # Diagonal indices.
  ilo = np.tril_indices(len(resolutions), -1)  # Lower triangle without diagonal indices.
  
  mean_byExp_up = np.round([np.mean(means[iup]) for means in data], decimals)
  mean_byExp_di = np.round([np.mean(means[idi]) for means in data], decimals)
  mean_byExp_lo = np.round([np.mean(means[ilo]) for means in data], decimals)
  
  if verbose==True:
    for i, mean_up in enumerate(mean_byExp_up):
      mean_di = mean_byExp_di[i]
      mean_lo = mean_byExp_lo[i]
      print(f'Mean fidelity with exponent {exponents[i]} upper {mean_up}, diagonal {mean_di}, lower {mean_lo}')
  
  return [mean_byExp, mean_byExp_up, mean_byExp_di, mean_byExp_lo]

means_tri_meth1_rel = means_withTriangles(means_meth1_r, resolutions, exponents)
means_tri_meth2_rel = means_withTriangles(means_meth2_r, resolutions, exponents)








