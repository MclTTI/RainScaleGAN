"""
Compute and plot the power spectrum
of a given numpy array
"""


import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras



def powerspec(a):
    """Compute the spatial power spectrum, averaged over time.
    
    Parameters:
        a : numpy ndarray
            Input array, with shape (time,lon,lat).
            
    Returns:
        Abins: numpy ndarray
               Average squared modulus of the Fourier amplitude in each k bin 
        kvals : numpy ndarray
                Norm of wave vectors k (pixel frequency)
    """
    
    #print(type(a))
        
    #Check if the array is square
    if a.shape[1] != a.shape[2]:
        print("The array have to be square!")
        return None
    
    #Compute the square modulus of Fourier amplitudes    
    npix = a.shape[1]
    af = np.abs(np.fft.fftn(a,axes=(1,2)) / (npix * npix))**2
    #ak = np.fft.fftn(a,axes=(1,2))
    
    #for t in range(af.shape[0]):
       #print(af[1,t,:])
    #   af_c = af
       #af_c[:,0,0]=0
       #print(int(af.shape[1]/2+1))
    #   print(np.sum(af_c[t,0:int(af.shape[1]/2+1),0:int(af.shape[1]/2+1)])*2)
    #   print(np.var(a[t,:,:]))
    #print(af.shape)
    af = np.mean(af,axis=0)
    #af_out = np.sum(af)
        
    #Compute the wave vector array
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    
    #Flatten the wave vector norm and the Fourier amplitudes
    knrm = knrm.flatten()
    af = af.flatten()
    
    #Bin the amplitudes in the k-space
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm,af,statistic = "mean",bins = kbins)
    
    #Compute the total power in each bin
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)     # Equivalent to 2pi k dk, equal when dk=1
    #Abins *= 2 * np.pi * kvals                         # Equivalent to 2pi k dk, equal when dk=1
    return Abins, kvals



def compute_lsd(Af,Af_ref):

    log10 = tf.experimental.numpy.log10

    Af_ref_db = 10 * log10(Af_ref)
    Af_db = 10 * log10(Af)

    Af_MSE = keras.metrics.mean_squared_error(Af_ref_db,Af_db)
    lsd = tf.math.sqrt(Af_MSE)

    return lsd



def plotspec_comparison(*spec,epoch=None,run=None):

    fig, ax = plt.subplots(figsize=(7,7))

    for i,s in enumerate(spec):

        #Extract power spectrum and reference power spectrum
        Af, k, Af_ref, k_ref, label, label_ref, color, color_ref = s

        #Initialize ref spectrum
        if i == 0:
            Af_ref_save = np.zeros(Af_ref.shape)
            k_ref_save = np.zeros(k_ref.shape)

        #Check if reference power spectrum changed
        if tf.math.reduce_all(tf.equal(Af_ref_save,Af_ref)) and tf.math.reduce_all(tf.equal(k_ref_save,k_ref)):
            plot_ref = False
        else:
            plot_ref = True

        #Update reference power spectrum
        Af_ref_save, k_ref_save = Af_ref, k_ref
        
        # Compute Log Spectral Distance
        lsd = compute_lsd(Af,Af_ref)

        # Plot reference spectrum if not already plotted
        if plot_ref == True:
            ax.loglog(k_ref,Af_ref,
                      label=label_ref,
                      color=color_ref,
                      linewidth=2)
        
        # Plot spectrum
        ax.loglog(k,Af,
                  label=f'{label} (Log-spectral dist. = {lsd:.4f})',
                  color=color,
                  linewidth=2)
         
    # Title
    fig.suptitle(f'Average spatial power spectrum (norm.)',fontsize='x-large')
    if epoch != None:
        ax.set_title(f'Epoch {epoch:003}',fontsize='large',loc='right')
    if run:
        ax.set_title(f'Run: {run}',fontsize='large',loc='left')
    
    # Labels, grid, legend
    ax.set_xlabel('$k$',fontsize='large')
    ax.set_ylabel(r'$|A(k)|^2$',fontsize='large')
    
    ax.set_ylim([0.01,40])
    
    ax.xaxis.grid(True,which='both')
    ax.yaxis.grid(True,which='both')
    
    ax.legend()
    
    plt.tight_layout()
            
    return fig



def plotspec_comparison_valid(Af_T,k_T,
                              Af_T_ref,k_T_ref,
                              Af_V,k_V,
                              Af_V_ref,k_V_ref,
                              epoch,run):

    fig, ax = plt.subplots(figsize=(8,8))
    
    # Compute Log Spectral Distance
    lsd_T = compute_lsd(Af_T,Af_T_ref)
    lsd_V = compute_lsd(Af_V,Af_V_ref)

    # Plot spectrum
    ax.loglog(k_T,Af_T,label=f'GAN (Train) (Log-spectral dist. = {lsd_T:.4f})',color='red',lw=1.5)
    ax.loglog(k_V,Af_V,label=f'GAN (Valid) (Log-spectral dist. = {lsd_V:.4f})',color='red',ls='--',lw=1.5)

    ax.loglog(k_T_ref,Af_T_ref,label=f'ERA5 (Train)',color='blue',lw=1.5)
    ax.loglog(k_V_ref,Af_V_ref,label=f'ERA5 (Valid)',color='blue',ls='--',lw=1.5)
  
    # Title
    fig.suptitle(f'Average spatial power spectrum (norm.)',fontsize='x-large')
    if epoch != None:
        ax.set_title(f'Epoch {epoch:003}',fontsize='large',loc='right')
    if run:
        ax.set_title(f'Run: {run}',fontsize='large',loc='left')
    
    # Labels, grid, legend
    ax.set_xlabel('$k$',fontsize='large')
    ax.set_ylabel(r'$|A(k)|^2$',fontsize='large')
    
    ax.set_ylim([0.01,40])
    
    ax.xaxis.grid(True,which='both')
    ax.yaxis.grid(True,which='both')
    
    ax.legend()
    
    plt.tight_layout()
            
    return fig


