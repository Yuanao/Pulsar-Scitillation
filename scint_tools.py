import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pywt
from numba import jit
import itertools
import pandas as pd

#@jit()
def para_model(eta,x,y,sym,array,plot=False):
        y_num,x_num = array.shape
        x_the_l = -np.sqrt((y+eta*sym**4)/eta)+sym
        #print(sym)
        #print(x_the_l)
        #x_the_l = x_the_l[x_the_l>=min(x)]
        dx_l = int(len(x_the_l[x_the_l<min(x)]))
        #print(dx_l)
        x_the_r = np.sqrt((y+sym**4)/eta)+sym
        #x_the_r = x_the_r[x_the_r<=max(x)]
        dx_r = int(len(x_the_r[x_the_r>max(x)]))
        width = int(max(x)*0.001/(x[1]-x[0]))
        if width <1:
           width=1
        if dx_l*dx_r!=0:
           x_the_l = x_the_l[:-dx_l]
           x_the_r = x_the_r[:-dx_r]
           x_the = np.unique(np.array(list(x_the_l)+list(x_the_r)))
           y_ind = np.hstack((np.arange(len(y))[::-1],np.arange(len(y))))[dx_l:-dx_r][:len(x_the)]
           y_ind = y_ind.repeat(width) ###as x_ind=[x_ind-1,x_ind,x_ind+1]
        else:
           x_the = np.unique(np.array(list(x_the_l)+list(x_the_r)))
           y_ind = np.hstack((np.arange(len(y))[::-1],np.arange(len(y))))[:len(x_the)]
           y_ind = y_ind.repeat(width*2)
        x_ind=[]
        #print(x_the_r,x_the_l)
        for i in range(len(x_the)):
            x_inx=x_the[i]
            #y_the = eta*x_inx**2+sym*x_inx
            x_idx = np.where(np.abs(x-x_inx)<=np.min(np.abs(x-x_inx)))[0][0]
            x_idx_ = list(np.arange(x_idx-width,x_idx+width))
            x_ind.append(x_idx_)
        x_ind = np.array(list(itertools.chain.from_iterable(x_ind)))
        x_ind[x_ind<0]=0
        x_ind[x_ind>x_num-1]=x_num-1
        if plot:
           cp = array.copy()
           vmin = np.nanmedian(cp.reshape(-1))
           vmax = np.nanmax(cp.reshape(-1))
           cp[y_ind,x_ind] = np.nan
           #plt.pcolormesh(x,y,array,cmap='jet',vmin=np.median(array.copy().reshape(-1)),vmax=np.nanmean(array)+2*np.nanstd(array))
           plt.pcolormesh(x,y,cp,cmap='jet',vmin=vmin,vmax=vmax)
           #plt.scatter(x[x_ind],y[y_ind],color='black',s=20) 
           plt.colorbar()
           plt.xlabel('$f_{t}$ (mHz)')
           plt.ylabel('$f_{\mu}$ (us)')
           plt.show()
        else:
           power_sum = np.nanmean(array[y_ind,x_ind])
           return power_sum

def svd_model(arr, nmodes=1):
    u, s, w = np.linalg.svd(arr)
    s[nmodes:] = 0
    S = np.zeros([len(u), len(w)], np.complex128)
    S[:len(s), :len(s)] = np.diag(s)
    model = np.dot(np.dot(u, S), w)
    arr = arr / np.abs(model)
    return arr


@jit()
def trans_model(array,x,y,curvature,symmetry,norm=2,plot=False):
         if symmetry==0:
            new_array=array
         else:
            x_start = int(symmetry)
            new_array = np.vstack((array[x_start:,:],array[:x_start,:]))
         x_start_r = np.sqrt(y/curvature)-norm
         x_start_l = -np.sqrt(y/curvature)+norm
         dx=x[1]-x[0]
         ny = len(y)   
         nx = len(x)
         trans_array=[]
         for i in range(ny):
            #if np.isnan(self.sspec[i]).any()==True:
            if np.isnan(new_array[i][0]):
              trans_array.append(new_array[i]) 
            else:
               array_l = new_array[i][0:nx//2]
               array_r = new_array[i][nx//2:]                     
               trans_sca_l = int(x_start_l[i]/dx)
               if abs(trans_sca_l)>nx//2:
                  trans_sca_l = np.sign(trans_sca_l)*nx//2  
               nan_array_l = np.zeros(abs(trans_sca_l))
               nan_array_l[:] = np.nan
               trans_sca_r = int(x_start_r[i]/dx)
               if abs(trans_sca_r)>nx//2:
                  trans_sca_r = np.sign(trans_sca_r)*nx//2
               nan_array_r = np.zeros(abs(trans_sca_r))
               nan_array_r[:] = np.nan
               if trans_sca_l <0:
                  new_array_l0 = list(nan_array_l)+list(array_l[:trans_sca_l])
               elif trans_sca_l ==0:
                  new_array_l0= list(array_l)
               else:
                  new_array_l0 = list(array_l[trans_sca_l:])+list(nan_array_l)
               if trans_sca_r <0:
                  new_array_r0 = list(nan_array_r)+list(array_r[:trans_sca_r])
               elif trans_sca_r ==0:
                  new_array_r0= list(array_r)
               else:
                  new_array_r0 = list(array_r[trans_sca_r:])+list(nan_array_r)
               trans_array0 = new_array_l0+new_array_r0
               trans_array.append(trans_array0)
         trans_array = np.array(trans_array)
         mean_power = np.nansum(trans_array,axis=0)
         #edge = max(x)/10
         edge=0.2
         sum_power_l = np.nansum(mean_power[(x>-norm-edge)&(x<-norm+edge)])
         sum_power_r = np.nansum(mean_power[(x>norm-edge)&(x<norm+edge)])
         sum_power= sum_power_l+sum_power_r
         if plot:
            print(edge,sum_power)
            cp = trans_array.reshape(-1)
            ax1=plt.subplot2grid((6,1),(0,0),rowspan=2)
            ax1.plot(x,mean_power)
            #ax1.scatter(x[(x>-norm-0.5)&(x<-norm+0.5)],mean_power[(x>-norm-0.5)&(x<-norm+0.5)],color='black')
            #ax1.scatter(x[(x>norm-0.5)&(x<norm+0.5)],mean_power[(x>norm-0.5)&(x<norm+0.5)],color='black')
            ax1.set_xlim(min(x),max(x))
            ax1.set_xticklabels([])
            ax1.set_ylabel('The weighted power (dB)')
            ax2=plt.subplot2grid((6,1),(2,0),rowspan=4)
            ax2.pcolormesh(x,y,trans_array,cmap=plt.get_cmap('jet'),vmin=np.nanmedian(cp),vmax=np.nanmax(cp),rasterized=True)
            ax2.set_xlabel('$f_{t}$ (mHz)')
            ax2.set_ylabel('$f_{\mu}$ ($\mu$s)')
            ##plt.imshow(trans_array,aspect='auto',origin='lower',cmap='jet',vmin=0,vmax=10)
            #plt.colorbar()
            plt.subplots_adjust(left=0.15,bottom=0.1,top=0.9,right=0.95,hspace=0,wspace=0)
            plt.show()
         return sum_power




def wavelet(sig,threshold = 3, level=3, wavelet='db8'):
    sig = np.array(sig)
    sigma = sig.std()
    dwtmatr = pywt.wavedec(data=sig, wavelet=wavelet, level=level)
    denoised = dwtmatr[:]
    denoised[1:] = [pywt.threshold(i, value=threshold*sigma, mode='soft') for i in dwtmatr[1:]]
    smoothed_sig = pywt.waverec(denoised, wavelet, mode='sp1')[:sig.size]
    noises = sig - smoothed_sig
    return smoothed_sig,noises

def wavelet_fill(sig,threshold = 3, level=3, wavelet='db8'):
    idxarr = np.arange(len(sig))
    sig = np.array(sig)
    sigma = sig.std()
    dwtmatr = pywt.wavedec(data=sig, wavelet=wavelet, level=level)
    denoised = dwtmatr[:]
    denoised[1:] = [pywt.threshold(i, value=threshold*sigma, mode='soft') for i in dwtmatr[1:]]
    smoothed_sig = pywt.waverec(denoised, wavelet, mode='sp1')[:sig.size]
    noises = sig - smoothed_sig
    idxbad = idxarr[(noises >= 2*np.nanstd(noises)+np.nanmean(noises))|(noises <= -2*np.nanstd(noises)+np.nanmean(noises))]
    sig[idxbad] = smoothed_sig[idxbad] 
    return sig

###This is a function to find the potential peaks of the array. The original array(or signal) will be smoothed by wavelet, the "level" parameter expalins how smmothing the final signal is; the "prominence" is the measurement of the sharpness of the peak; "distance" is the minimum horizional distance between two peaks.###
def detect_peaks(sig,threshold=1,level=5,prominence=200,distance=0):
    ###use wavelet to smmoth the curve###
    sigma = sig.std()
    dwtmatr = pywt.wavedec(data=sig, wavelet=wavelet, level=level)
    denoised = dwtmatr[:]
    denoised[1:] = [pywt.threshold(i, value=threshold*sigma, mode='soft') for i in dwtmatr[1:]]
    smoothed_sig = pywt.waverec(denoised, wavelet, mode='sp1')[:sig.size]
    peaks_index = find_peaks(sig,prominence=200,distance=distance)[0] 
    
def skkt(array,edge_l,edge_r,step):
    test_data = pd.Series(array)
    sk = test_data.skew()
    kt = test_data.kurt()
    return sk,kt

def find_nearest(xarray, yarray,value,r=20):
    xarray = np.asarray(xarray)
    idx = (np.abs(xarray - value)).argmin()
    idx_arr = np.arange(idx-r,idx+r)
    value_new = max(yarray[idx_arr])
    yarray = np.asarray(yarray)
    idx_new = (np.abs(yarray - value_new)).argmin()
    return xarray[idx_new]

### This is a function to fitting the peak and its neibours with gaussian funciton. Before fitting, the suitable x-axis range should be decided. The original distance between the edge and peak is set as "distance=50". In case to   
def array_gaus(x,y,y_smooth,peak_index,distance=500,Fitting=False):
    from scipy import optimize
    from math import sqrt
    num = len(x)
    x_l0 = peak_index-distance
    x_r0 = peak_index+distance
    if x_l0 < 0:
       x_l0 = 0
    if x_r0 > num+distance:
       x_r0 = num
    #y_diff_l = y[x_l0:peak_index]-y[peak_index]
    #y_diff_r = y[peak_index+1:]-y[peak_index]
    y_diff = np.diff(y_smooth)
    y_diff = np.insert(y_diff,peak_index,0)
    y_diff_l = y_diff[x_l0:peak_index]
    y_diff_r = y_diff[peak_index:x_r0]
    y_diff_l_ = y_diff_l>=0
    y_diff_r_ = y_diff_r<=0 
    if y_diff_l_.all() == True:
       x_l=x_l0
    else:
       x_l=peak_index-np.where(y_diff_l[::-1]<0)[0][0]-1
    if y_diff_r_.all() == True:
       x_r=x_r0
    else:
       x_r=peak_index+np.where(y_diff_r>0)[0][0]+1
    #print(x_l,x_r)
    x_fit = x[x_l:x_r]
    #print(x_fit)
    if Fitting:
       y_fit = y_smooth[x_l:x_r]
    else:
       y_fit=y[x_l:x_r]
    x_fit_norm=(x_fit-min(x_fit))/(max(x_fit)-min(x_fit)) 
    y_fit_norm=(y_fit-min(y_fit))/(max(y_fit)-min(y_fit)) 
    mu =  x_fit[y_fit_norm>=max(y_fit_norm)][0]
    print(mu)      
    def Gaussian(x_fit_norm,a,b,c):
         return a* np.exp(-(x_fit_norm - mu +b)**2.0 / (2 * c**2))
    #def Gaussian(x_fit_norm,a,b,c):
    #    return a*(x_fit_norm-mu+b)**c+a*b**c
    error = []
    popt, pcov = optimize.curve_fit(Gaussian,x_fit_norm,y_fit_norm)
    perr = np.sqrt(np.diag(pcov))
    for i in range(len(popt)):
        try:
            error.asppend(np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    pfit_curvefit = popt
    #pfit_curvefit[1]=0
    perr_curvefit = np.array(error)
    yerr=Gaussian(x_fit,*pfit_curvefit)
    y_gauss = Gaussian(x_fit_norm, *popt) 
    #best_mu = popt[1]*(max(x_fit)-min(x_fit))+min(x_fit)
    best_mu = mu
    best_error = popt[1]*(max(x_fit)-min(x_fit))
    newx = x_fit
    newy=y_gauss*(max(y_fit)-min(y_fit))+min(y_fit)
    return best_mu,best_error,newx,newy
    '''
    print('\n The bset curvature from curve-fit: %s' % best_mu)
    print('\n The errors: %s' % best_error)
      # Plot the fit
    plt.plot(x_fit_norm*(max(x_fit)-min(x_fit))+min(x_fit), y_gauss*(max(y_fit)-min(y_fit))+min(y_fit),linestyle='--',linewidth=2,color='red',label='Gaussian')
    plt.plot(x_fit_norm*(max(x_fit)-min(x_fit))+min(x_fit),y_fit_norm*(max(y_fit)-min(y_fit))+min(y_fit)
,linewidth=2,color='black')
    plt.show()
    '''
