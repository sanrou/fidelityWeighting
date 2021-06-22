# -*- coding: utf-8 -*-
"""
Phase visualization for fidelity weighting.

@author: rouhinen
"""


### Phase visualization
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from scipy import signal


"""Load source identities, forward and inverse operators from csv. """
dataPath = 'C:\\temp\\fWeighting\\csvSubjects_p\\sub (5)'
# dataPath = 'K:\\palva\\fidelityWeighting\\csvSubjects_p\\sub (5)'

fileSourceIdentities = glob.glob(dataPath + '\\*parc68.csv')[0]
fileForwardOperator  = glob.glob(dataPath + '\\*forwardOperatorMEEG.csv')[0]
fileInverseOperator  = glob.glob(dataPath + '\\*inverseOperatorMEEG.csv')[0]

delimiter = ';'
identities = np.genfromtxt(fileSourceIdentities, 
                                    dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                                    dtype='float', delimiter=delimiter))        # sensors x sources
inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                                    dtype='float', delimiter=delimiter))        # sources x sensors

""" Settings """
n_samples = 200
n_cut_samples = 20
widths = range(5,6)
tightLayout = False

""" Get number of parcels. """
idSet = set(identities)                         # Get unique IDs
idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
n_parcels = len(idSet)

""" Generate parcel signal """
np.random.seed(41)
s = randn(n_parcels, n_samples+2*n_cut_samples)

for i in np.arange(0, n_parcels):
    s[i, :] = signal.cwt(s[i, :], signal.ricker, widths)

s = signal.hilbert(s)
parcelSeries = s[:, n_cut_samples:-n_cut_samples]

""" Parcel series to source series. 0 signal for sources not belonging to a parcel. """
sourceSeries = parcelSeries[identities]
sourceSeries[identities < 0] = 0

""" Forward then inverse model source series. """
# Values wrapping? Forward and inverse operators norm to 1 to avoid this.
forward = forward / norm(forward)
inverse = inverse / norm(inverse)

sourceSeries = np.dot(inverse, np.dot(forward, sourceSeries))

# Normalize amplitude.
parcelSeries = np.einsum('ij,i->ij', parcelSeries, 1/np.max(abs(parcelSeries), axis=1))   # Value error with this one for some reason with a normal divide.
sourceSeries = sourceSeries / np.max(sourceSeries, axis=1)

"""  Plot  """
# parcels = [0, 0, 0] # Same parcel to use on the phase plot ring. 
# sources = [985, 1255, 1541]   # Parcel0
parcels = [14, 14, 14] # Same parcel to use on the phase plot ring. 
sources = [386, 473, 852]  # Parcel14. 386 ~0.85 PLV. 473 ~0.65 PLV, flip. 852 ~0.26 PLV.
# colors = ['black', 'dimgray', 'brown']
colors = ['black', 'black', 'black']

# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
if tightLayout == True:
  params = {'legend.fontsize':'7',
         'figure.figsize':(4, 1.3),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.sans-serif':'Arial',
         'lines.markersize':3}
else:
  params = {'legend.fontsize':'7',
         'figure.figsize':(11, 3),
         'axes.labelsize':'14',
         'axes.titlesize':'14',
         'xtick.labelsize':'14',
         'ytick.labelsize':'14',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.sans-serif':'Arial',
         'lines.markersize':6}
pylab.rcParams.update(params)


""" Plot time series and phases """
fig = plt.figure()

for i in range(len(parcels)):
    ax = fig.add_subplot(1, 3, i+1)
    ax.plot(np.real(parcelSeries[parcels[i],:].T)+2.2, color=colors[i], label='Parcel ' + str(parcels[i]))
    ax.plot(np.real(sourceSeries[sources[i],:].T), color=colors[i], label='Source ' + str(sources[i]))
    
    ax.plot(np.angle(parcelSeries[parcels[i],:].T)/3.14 -2.8, color=colors[i])
    ax.plot(np.angle(sourceSeries[sources[i],:].T)/3.14 -5, color=colors[i])
    
    ax.plot(np.ravel(np.angle(np.exp(1j*(np.angle(parcelSeries[parcels[i],:]) 
                            - np.angle(sourceSeries[sources[i],:])))))/3.14 -7.2, color=colors[i])     # Angle difference
    
    ax.set_ylabel('Real or Phase')
    ax.set_xlabel('Time')
    # plt.axis('off')
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()
    
plt.tight_layout(pad=0.1)
plt.show()


""" Plot angle difference phase ring. """
fig = plt.figure()

for i in range(len(parcels)):
    ax = fig.add_subplot(1, 3, i+1)
    
    # Subtract original and modeled phases.
    subt = np.exp(1j*(np.angle(parcelSeries[parcels[i],:]) - np.angle(sourceSeries[sources[i],:])))
    
    # Plot phase differences of samples as angle ring.
    ax.plot(np.real(subt.T), np.imag(subt.T), 'ko', alpha=0.2, color=colors[i]) # , fillstyle='none'
    
    # Draw arrow to the mean from origin.
    arrowX = np.real(np.mean(subt))
    arrowY = np.imag(np.mean(subt))
    ax.arrow(0, 0, arrowX, arrowY, shape='full', width=0.02, color='k', length_includes_head=True)  # cPLV
    ax.arrow(0, 0, arrowX, 0, shape='full', width=0.02, color='dimgray', length_includes_head=True) # real(cPLV)
    
    # Set axis invisible top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_ylabel('Imaginary')
    ax.set_xlabel('Real, PLV = {:.3f}'.format(abs(np.mean(subt))))
    # plt.axis('off')
    
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    
plt.tight_layout(pad=0.1)
plt.show()


