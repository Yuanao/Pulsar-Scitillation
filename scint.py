import time
import os
import pylab
from os.path import split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.constants as sc
from copy import deepcopy as cp
from scipy import stats
from scipy.interpolate import griddata, interp1d, RectBivariateSpline
from scipy.signal import convolve2d, medfilt, savgol_filter,find_peaks
from scipy.io import loadmat
from lmfit import Parameters
from lmfit import Minimizer, conf_interval
import pandas as pd
import sys
import warnings
from scint_tools import para_model,trans_model,wavelet,array_gaus,svd_model,wavelet_fill,find_nearest
 
warnings.filterwarnings('ignore')

params={'axes.labelsize': '20',
                  'xtick.labelsize':'15',
                 'ytick.labelsize':'15',
                 'lines.linewidth':'2' ,
                 'legend.fontsize': '20',
                 }
pylab.rcParams.update(params)

class Sci:
     def __init__(self):
        """"
        Initialise a dynamic spectrum object by either reading from file
            or from existing object
        """
        self.df = None
        self.freqs = None
        self.freq = None
        self.nchan = None
        self.dt = None
        self.dsfile = None
        self.dynfile = None
        self.dyn = None
        self.sspec = None
        
     def load_dyn_obj(dynfile,self):
        if self.dynfile:
           self.dyn = np.loadtxt(dynfile)
        else:
           print('Please input the dynamic spectrum!')
        if self.df and self.dt and self.freqs:
           print(df,dt,freqs)
        else:
           print('Please input the df,dt, and frequency!')
        #return self.dyn, self.dt, self.df,self.freq

     def load_file(self,dsfile,fmin=False,fmax=False):
        print("LOADING {0}...".format(dsfile))
        head = []
        with open(dsfile, "r") as file:
             for line in file:
                 if line.startswith("#"):
                    headline = str.strip(line[1:])
                    head.append(headline)
                 if str.split(headline)[0] == 'MJD0:':
                    # MJD of start of obs
                    mjd = float(str.split(headline)[1])
        self.name = os.path.basename(dsfile)
        self.basename = self.name.split('.')[0]
        self.header = head
        rawdata = np.loadtxt(dsfile).transpose()  # read file
        self.times = np.unique(rawdata[2]*60)  # time since obs start (secs)
        self.freqs = rawdata[3]  # Observing frequency in MHz.
        fluxes = rawdata[4]  # fluxes
        fluxerrs = rawdata[5]  # flux errors         
        self.nchan = int(np.unique(rawdata[1])[-1])  # number of channels
        self.bw = self.freqs[-1] - self.freqs[0]  # obs bw
        self.df = round(self.bw/self.nchan, 5)  # channel bw
        self.bw = round(self.bw + self.df, 2)  # correct bw
        self.nchan += 1  # correct nchan
        print('number of channels:%s'%self.nchan)
        self.nsub = int(np.unique(rawdata[0])[-1]) + 1
        self.tobs = self.times[-1]+self.times[0]  # initial estimate of tobs
        self.dt = self.tobs/self.nsub
        if self.dt > 1:
            self.dt = round(self.dt)
        else:
            self.times = np.linspace(self.times[0], self.times[-1], self.nsub)
        print('*The integration time: %s s.'%self.dt)
        self.tobs = self.dt * self.nsub  # recalculated tobs
        # Now reshape flux arrays into a 2D matrix
        self.freqs = np.unique(self.freqs)
        f_copy = cp(self.freqs)
        fluxes = fluxes.reshape([self.nsub, self.nchan]).transpose()
        fluxerrs = fluxerrs.reshape([self.nsub, self.nchan]).transpose()
        if self.df < 0:  # flip things
            self.df = -self.df
            self.bw = -self.bw
            # Flip flux matricies since self.freqs is now in ascending order
            fluxes = np.flip(fluxes, 0)
            fluxerrs = np.flip(fluxerrs, 0)
        # Finished reading, now setup dynamic spectrum
        ####frequency choice#####
        if fmin and fmax:
           if fmax <= min(self.freqs):
               print('*Please input the right frequency between %3.f ~ %3.f MHz!!!'%(min(self.freqs),\
                                                                                   max(self.freqs)))
               sys.exit(0)
           self.freqs = self.freqs[(f_copy >= fmin) & ((f_copy <= fmax))]
           self.freq = round(np.mean(self.freqs), 2)
           self.dyn = fluxes[(f_copy >= fmin) & ((f_copy <= fmax))]
        elif fmin or fmax:
           print('Please input the minimum AND the maximum frequency!!!')
           sys.exit(0) 
        else:
           self.dyn = fluxes
        #return self.dyn, self.dt, self.df,self.freq  
        print('*The frequency band: %.3f ~ %.3f MHz'%(min(self.freqs),max(self.freqs)))
        print('*The frequency sampling:%s MHz.'%self.df)
        return self.dyn
        
     def pro_dyn(self,model='svd',tem=True,fre=True):
         if model=='wavelet':
            print('\nSMOOTHING THE DYNMIC SPECTRUM BY WAVELET...')
            if tem:
               tem_w=[]  
               for i in range(self.dyn.shape[0]):  
                   temporal_s = self.dyn[i]
                   g=wavelet_fill(temporal_s)
                   tem_w.append(g)
               self.dyn = np.array(tem_w) 
            if fre:
               fre_w=[]
               for i in range(self.dyn.shape[1]):
                   fre_s = self.dyn[:,i]
                   g=wavelet_fill(fre_s)
                   fre_w.append(g)
               self.dyn = np.array(fre_w).T
         else:
            print('\nSMOOTHING THE DYNMIC SPECTRUM BY SVD...')
            self.dyn = svd_model(self.dyn)
         return self.dyn   
            
     def get_sspec(self,fdop_del=15):
        print('\n')
        print('CALCULATING THE SECONDARY SPECTRUM ...')
        self.fdop_del = fdop_del
        nf = np.shape(self.dyn)[0]
        nt = np.shape(self.dyn)[1]
        dyn_norm = self.dyn - np.mean(self.dyn)
        nrfft = int(2**(np.ceil(np.log2(nf))+1))
        ncfft = int(2**(np.ceil(np.log2(nt))+1))
        simf = np.fft.fft2(dyn_norm,s=[nrfft, ncfft])

        simf = np.real(np.multiply(simf, np.conj(simf)))  # is real
        sec = np.fft.fftshift(simf)  # fftshift
        sec = sec[int(nrfft/2):][:]  # crop
        td = np.array(list(range(0, int(nrfft/2))))
        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        self.fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        self.tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)]) #in us
        self.dfdop = self.fdop[1]-self.fdop[0]
        self.dtel = self.tdel[1]-self.tdel[0]
        sec = np.log(sec)
        self.sec_raw =sec.copy()
        sec[:, int(len(self.fdop)/2-np.floor(4/2)):int(len(self.fdop)/2+np.ceil(4/2))+1] = np.nan
        sec[:fdop_del, :] = np.nan
        self.sspec = sec
            
     def find_sym(self,plot=False,tdel_ign = 0,f_stat=-1,f_end=1):
         print('\n')
         print('FINDING THE SYMMETRY AXIS ...')    
         from scipy.optimize import curve_fit
         self.fdop = np.array(self.fdop)
         sym_arr = self.fdop[(self.fdop>=f_stat) & (self.fdop <=f_end)]
         #sym_arr = self.fdop[len(self.fdop)//2-100,len(self.fdop)//2+100]
         sspec_power_re=[]
         for sym in sym_arr:
             edge = list(self.fdop).index(sym)
#             sspec_left = self.sspec[tdel_ign:,:edge]
#             sspec_right = self.sspec[tdel_ign:,edge:]
             sspec_left = self.sec_raw[tdel_ign:,:edge]
             sspec_right = self.sec_raw[tdel_ign:,edge:]
             sspec_left_sum = np.nansum(sspec_left,axis=1)
             sspec_right_sum = np.nansum(sspec_right,axis=1)
             sspec_re  = abs(np.nansum(sspec_left_sum - sspec_right_sum))
             sspec_power_re.append(sspec_re)
         sspec_power_re = np.array(sspec_power_re)
         def func(x,a,b):
             return a*x+b
#         def func(x,a,b,c,d,e,f):
#             return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f
         popt_l, pcov_l = curve_fit(func, sym_arr[sym_arr<0], sspec_power_re[sym_arr<0])
         popt_r, pcov_r = curve_fit(func, sym_arr[sym_arr>0], sspec_power_re[sym_arr>0])
#         sym_arr_new = np.linspace(-5,5,2*len(self.fdop))
         sym_arr_l = np.linspace(min(sym_arr),max(sym_arr),2*len(sym_arr))
         sym_arr_r = np.linspace(min(sym_arr),max(sym_arr),2*len(sym_arr))
#         sym_arr_l0 = np.linspace(min(sym_arr),0.5,2*len(sym_arr))
#         sym_arr_r0 = np.linspace(-0.5,max(sym_arr),2*len(sym_arr))
         sym_re_l = func(sym_arr_l, *popt_l)
         sym_re_r = func(sym_arr_r, *popt_r)
         re = abs(sym_re_l-sym_re_r)
#         self.symmetry = sym_arr[(sym_arr<=0.5)&(sym_arr>=-0.5)][abs(sym_cro_l-sym_cro_r)<=min(abs(sym_cro_l-sym_cro_r))]
         sym = sym_arr_l[np.where(re == min(re))[0][0]]
         self.symmetry = sym 
         
         if plot:
            print("*the symmetry axis is at ETA = %.3f (mHz)"%self.symmetry)
            plt.scatter(sym_arr,sspec_power_re,color='saddlebrown',label = 'The summed-power')
            plt.plot(sym_arr_l[sym_arr_l<=sym],sym_re_l[sym_arr_l<=sym],color='black',label = 'The fitting summed-power')
            plt.plot(sym_arr_r[sym_arr_r>=sym],sym_re_r[sym_arr_r>=sym],color='black')
            plt.text(self.symmetry+0.8,0.9*max(sym_re_r),"Symmetry:%.3f"%self.symmetry,fontsize=20)
#            plt.plot(sym_arr[sym_arr<=self.symmetry],sym_re_l[sym_arr[sym_arr<=0.5]<=self.symmetry],
#                     color='black',label = 'The fitting summed-power')
#            plt.plot(sym_arr[sym_arr>=self.symmetry],sym_re_r[sym_arr[sym_arr>=-0.5]>=self.symmetry],color='black')
            plt.axvline(self.symmetry,linestyle='--',color='blue', label='The fitting symmetry axis')
            plt.legend(loc='best')
            plt.ylabel('Summed power (dB)')
            plt.xlabel('Symmetry axis (mHz)')
            plt.show()
         else:
            print("*If you want to plot the figure, please input 'plot=True'.")
         return self.symmetry 

     def arc_trans(self,eta_min=0.05,eta_max=2,eta_step=5000,level=1,prominence=200,plot_model=True,
                   plot_power=False,load_power=False,kendalltau=True):
         print('\n')
         print('CALCULATING THE SUMMED POWER...')
         symmetry = int(self.symmetry/self.dfdop)
         if load_power:
             try:
                eta_p = np.loadtxt('power-trans-%s.txt'%self.basename)
                print('*Loading weighted-power from power-trans-%s.txt...'%self.basename)
                eta_array = eta_p[0]
                weighted_power = eta_p[1]
                print('*The ETA range:%s ~ %s;'%(min(eta_array),max(eta_array)))
                print('*The ETA setp:%s.'%len(eta_array))
             except:
                print("No model has been calculated! \
                      You can calculate the models like 'scint.arc_trans()' or 'scint.arc_para()'.")
         else:    
             print('*The ETA range:%s ~ %s;'%(eta_min,eta_max))
             print('*The ETA setp:%s.'%eta_step)
             weighted_power=[]
#             eta_array1=np.linspace(eta_min,1,2*eta_step//2)
#             eta_array2=np.linspace(1,eta_max,eta_step//2)
#             eta_array = np.hstack((eta_array1,eta_array2))
             eta_array = np.linspace(eta_min,eta_max,eta_step)
             count=0
             for eta in eta_array:
                 count+=1
                 if count==1:
                    print('*To check the secondary spectrum has the right shape:')
                    print('-------The fdop number is %s'%self.sspec.shape[1])
                    print('-------The tdel number is %s'%self.sspec.shape[0])
                 print("\r*Calculation Finished:{}%".format(round(count*100/eta_step)), end="")
             
                 p=trans_model(self.sspec,self.fdop,self.tdel,eta,symmetry)
                 weighted_power.append(p)  
             array_s = np.vstack((np.array(eta_array),np.array(weighted_power)))
             np.savetxt('power-trans-%s.txt'%self.basename,array_s)
         sig, nos = wavelet(weighted_power,threshold = 1, level = level)
         eta_index = find_peaks(sig,prominence=prominence,distance=0.1*eta_step)[0]
         if len(eta_index)==0:
             print('###No peaks found!!!###')
             plt.plot(eta_array,weighted_power,linewidth=2,color='k',label='The weighted-total-power')
             plt.plot(eta_array,sig,'--',linewidth=2,color='red',label='The fitting weighted-total-power')
             plt.show()
             os._exit(0)
         else:    
             print('\n*The potential ETA is:%s.'%eta_array[eta_index])
         if kendalltau:
            eta_index_l = eta_index-int(0.03*eta_step)
            eta_index_l[eta_index_l<0]=0
            eta_index_r = eta_index+int(0.03*eta_step)
            eta_index_r[eta_index_r>len(sig)]=len(sig)
            inc_t=[]
            inc_p=[]
            for i in range(len(eta_index)):
                inc_t0,inc_p0 = stats.kendalltau(np.arange(eta_index_r[i]-eta_index_l[i]),sig[eta_index_l[i]:eta_index_r[i]])
                inc_t.append(inc_t0)
                inc_p.append(inc_p0)
            inc_t = np.array(inc_t)
            inc_p = np.array(inc_p)
         #eta_index0=eta_index[(abs(inc_t)<=0.9)|(abs(inc_p/inc_t)<=5)]
            eta_index=eta_index[abs(inc_t)<=0.8]
            inc_p = inc_p[abs(inc_t)<=0.8]
            inc_t = inc_t[abs(inc_t)<=0.8]
            p_t = np.array(abs(inc_p/inc_t))
            eta_index=eta_index[p_t<=1]
         if plot_model:
#             trans_model(self.sspec,self.fdop,self.tdel,0.087,symmetry,plot=True)
            trans_model(self.sspec,self.fdop,self.tdel,eta_array[eta_index[0]],symmetry,plot=True)
         else:
            print("*If you want to plot the fitting model, please input 'plot_model=True'.")
         if plot_power:
            plt.plot(eta_array,weighted_power,linewidth=2,color='k',label='The weighted-total-power')
            plt.plot(eta_array,sig,'--',linewidth=2,color='red',label='The fitting weighted-total-power')
            plt.scatter(eta_array[eta_index],sig[eta_index],s=200,marker='d',color='blue',label = 'The bset curvature')
            plt.xlabel('$\eta (s^{3})$')
            plt.ylabel('The weighted-power (dB)')
            plt.legend(loc='best')
            plt.show()
         else:
            print("*If you want to plot the fitting power, please input 'plot_power=True'.")
         self.weighted_power = weighted_power
         self.eta_array = eta_array
         self.smoothed_power = sig
         self.eta_goods = eta_array[eta_index]
        
         
     def arc_para(self,eta_min=0.05,eta_max=2,eta_step=5000,level=3,prominence=0,plot_model=False,plot_power=False,
                  load_power=False,kendalltau=True):
         print('\n')
         print('CALCULATING THE SUMMED POWER...')
         if load_power:
             try:
                eta_p = np.loadtxt('power-para-%s.txt'%self.basename)
                print('*Loading weighted-power from power-para-%s.txt...'%self.basename)
                eta_array = eta_p[0]
                weighted_power = eta_p[1]
                print('*The ETA range:%s ~ %s;'%(min(eta_array),max(eta_array)))
                print('*The ETA setp:%s.'%len(eta_array))
             except:
                print("#####No model has been calculated! You can calculate the models like 'scint.arc_trans()' or 'scint.arc_para()'.#####")
                os._exit(0)
         else:    
             print('*The ETA range:%s ~ %s;'%(eta_min,eta_max))
             print('*The ETA setp:%s.'%eta_step)
             weighted_power=[]
#         eta_array1=np.linspace(eta_min,1,2*eta_step//2)
#         eta_array2=np.linspace(1,eta_max,eta_step//2)
#         eta_array = np.hstack((eta_array1,eta_array2))
             eta_array = np.linspace(eta_min,eta_max,eta_step)
#         para_model(0.09,self.fdop,self.tdel,self.symmetry,self.sspec,plot=True)
             count=0
             for eta in eta_array:
                 count+=1
                 if count==1:
                    print('*To check the secondary spectrum has the right shape:')
                    print('-------The fdop number is %s'%self.sspec.shape[1])
                    print('-------The tdel number is %s'%self.sspec.shape[0])
                 print("\r*Calculation Finished:{}%".format(round(count*100/eta_step)), end="")
                 p=para_model(eta,self.fdop,self.tdel,self.symmetry,self.sspec)
                 weighted_power.append(p)
             array_s = np.vstack((np.array(eta_array),np.array(weighted_power)))
             np.savetxt('power-para-%s.txt'%self.basename,array_s) 
         sig, nos = wavelet(weighted_power,threshold = 0.5, level = level)
         eta_index = find_peaks(sig,prominence=prominence,distance=0.01*eta_step)[0]
         if len(eta_index)==0:
             print('###No peaks found!!!###')
             plt.plot(eta_array,weighted_power,linewidth=2,color='k',label='The weighted-total-power')
             plt.plot(eta_array,sig,'--',linewidth=2,color='red',label='The fitting weighted-total-power')
             plt.show()
             os._exit(0)
         else:    
             print('\n*The potential ETA is:%s.'%eta_array[eta_index]) 
         if kendalltau:
            eta_index_l = eta_index-int(0.03*eta_step)
            eta_index_l[eta_index_l<0]=0
            eta_index_r = eta_index+int(0.03*eta_step)
            eta_index_r[eta_index_r>len(sig)]=len(sig)
            inc_t=[]
            inc_p=[]
            for i in range(len(eta_index)):
                inc_t0,inc_p0 = stats.kendalltau(np.arange(eta_index_r[i]-eta_index_l[i]),sig[eta_index_l[i]:eta_index_r[i]])
                inc_t.append(inc_t0)
                inc_p.append(inc_p0)
            inc_t = np.array(inc_t)
            inc_p = np.array(inc_p)
         #eta_index0=eta_index[(abs(inc_t)<=0.9)|(abs(inc_p/inc_t)<=5)]
            eta_index=eta_index[abs(inc_t)<=0.8]
            inc_p = inc_p[abs(inc_t)<=0.8]
            inc_t = inc_t[abs(inc_t)<=0.8]
            p_t = np.array(abs(inc_p/inc_t))
            eta_index=eta_index[p_t<=1]
         if plot_model:
            if len(eta_index)==0:
                eta_m = 0.08
            else:
                eta_m=eta_array[eta_index[0]]
            para_model(eta_m,self.fdop,self.tdel,self.symmetry,self.sspec,plot=True)
         else:
            print("*If you want to plot the fitting model, please input 'plot_model=True'.")
         if plot_power:
            plt.plot(eta_array,weighted_power,linewidth=2,color='k',label='The weighted-total-power')
            plt.plot(eta_array,sig,'--',linewidth=2,color='red',label='The fitting weighted-total-power')
            plt.scatter(eta_array[eta_index],sig[eta_index],s=200,marker='d',color='blue',label = 'The bset curvature')
            #for j in range(len(eta_index)):
            #    plt.text(eta_array[eta_index[j]],sig[eta_index[j]],'(%.3f,%.3f)'%(inc_t[j],inc_p[j]),fontsize=20)
            plt.xlabel('$\eta (s^{3})$')
            plt.ylabel('The weighted-power (dB)')
            plt.legend(loc='best')
            plt.show()
         else:
            print("*If you want to plot the fitting power, please input 'plot_power=True'.")
         self.weighted_power = weighted_power
         self.eta_array = eta_array
         self.smoothed_power = sig
         self.eta_goods = eta_array[eta_index]
         

     def fit_arc(self,pick=None,distance=50,plot=True):
         print('\n')
         print('FITTING THE ARC...')
         if pick:
             if isinstance(pick,list) or type(pick) == np.ndarray:
                 if isinstance(pick[0],float):
                    self.eta_goods = np.array([find_nearest(self.eta_array,self.weighted_power,j) for j in pick])
                    print(self.eta_goods)
                 else:
                    self.eta_goods = np.array([self.eta_goods[i] for i in pick]).reshape(-1)
             else:
                 print("*Please input the 'pick' as a 'list' or 'array' type,like:'pick=[1,2,3,...]'.")
                 os._exit(0)
         else:
             print("*If you want to choose the fitting arc mannuly, please input the parameter like 'pick=[1,2,3,...]'.") 
         if plot:
             colspan_num = len(self.eta_goods)
             ax_0 = plt.subplot2grid((2,colspan_num), (1,0), colspan=colspan_num)
         i=-1
         best_arc=[]
         best_arc_e=[]
         for eta0 in self.eta_goods:
             i+=1
             eta0_index = int(np.where(self.eta_array==eta0)[0][0])
             best_mu,best_error,newx,newy=array_gaus(self.eta_array,self.weighted_power,
                                                     self.smoothed_power,eta0_index,distance=100,Fitting=True)
             print('#####The Best ETA:%s (error:%s)#####'%(best_mu,best_error))
             best_arc.append(best_mu)
             best_arc_e.append(best_error)
             self.best_arc = best_arc
             self.best_arc_e = best_arc_e
             if plot:
                ax=plt.subplot2grid((2,colspan_num), (0,i))
                ax.plot(newx,newy,color='blue',linewidth=2)
                ax_0.plot(newx,newy,color='blue',linewidth=2)
                ax_0.axvspan(best_mu-best_error, best_error+best_mu,color='lightsalmon',alpha=0.5)
                ax_0.axvspan(best_mu-best_error, best_error+best_mu,color='lightsalmon',alpha=0.5)
                ax.plot(newx,self.weighted_power[np.where(self.eta_array==min(newx))[0][0]:
                    np.where(self.eta_array==max(newx))[0][0]+1],color='k')
                ax.plot(newx,self.smoothed_power[np.where(self.eta_array==min(newx))[0][0]:
                    np.where(self.eta_array==max(newx))[0][0]+1],linestyle='--',color='red')
         if plot:
             ax_0.plot(self.eta_array,self.weighted_power,color='k')
             ax_0.plot(self.eta_array,self.smoothed_power,'--',color='red')
             ax_0.set_xlabel('$\eta (s^{3})$')
             ax_0.set_ylabel('The weighted power (dB)')
             plt.show()
         else:
             print("*If you want to plot the arc fitting result, please input 'plot=True'.")
            
     def plot_dyn(self,cm=False,save = False,**kw):
         try:
             cp = self.dyn.copy().reshape(-1)
         except:
             print('No dynamic spectrum found!! Please load file first!')
             os.exit(0)
        
         vmin = np.nanmedian(cp)
         #vmax = np.nanmax(cp)
         vmax = np.nanmean(cp)+3*np.nanstd(cp)
         plt.pcolormesh(self.times/60,self.freqs,self.dyn,cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax,rasterized=True)
         plt.xlabel('Time (min)')
         plt.ylabel('Frequency (MHz)')
         plt.tight_layout()
         if save:
             plt.savefig('%s_dyn.png'%self.name[:-3],dpi=300)
         plt.show()
         
     def plot_sspec(self,save=False,plot_arc=True):
         try:
             cp = self.sspec.copy().reshape(-1)
         except:
             print('No secondary spectrum found!! Please calculate the secondary spectrum first!')
             os.exit(0)
         vmin = np.nanmedian(cp)
         vmax = np.nanmax(cp)
         plt.pcolormesh(self.fdop,self.tdel,self.sspec,cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax,rasterized=True)
         plt.ylabel('$f_t$ ($\mu$s)')
         plt.xlabel('$f_{\mu}$ (mHz)')
         if plot_arc:
            try:
                for eta in self.best_arc:
                    f_model = eta*(self.fdop-self.symmetry)**2-eta*self.symmetry**2/4
                    plt.plot(self.fdop,f_model,'--',color='red',linewidth=1)
            except:
                print('Have not fit the arc. Please fit the arc first!')
                os._exit(0)
         plt.ylim(min(self.tdel),max(self.tdel))
         plt.tight_layout()
         if save:
             plt.savefig('%s_sspec.png'%self.name[:-3],dpi=300)
         plt.show()    
