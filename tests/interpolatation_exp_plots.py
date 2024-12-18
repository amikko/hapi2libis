#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:22:21 2024

@author: mikkonea
"""

import numpy as np
import netCDF4
import matplotlib.pyplot as plt

fc_folder = './uvspec_full_compute/'
bb_folders = ['./uvspec_interpolated_bb_coarse/',
              './uvspec_interpolated_bb_fine/',
              './uvspec_interpolated_bb_extrafine/',
              './uvspec_interpolated_bb_ultrafine/']
nn_folder = './uvspec_interpolated_nn/' 

vmaxs = [np.nan, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

n_data = 40
n_layer = 50
n_wn = 36765

mask_check = 1e-8

runtimes = np.zeros((n_data,6))
taus = np.zeros((n_data,n_layer,n_wn,6))
diffs = np.zeros((n_data * n_layer * n_wn,5))
ma = diffs < mask_check

for i in range(n_data):
    time_fname = 'time_%d.dat' % i
    taus_fname = 'uvspec_custom_%d.nc' % i 
    fc_time = np.genfromtxt(fc_folder + time_fname)
    nn_time = np.genfromtxt(nn_folder + time_fname)
    runtimes[i,0] = fc_time
    runtimes[i,1] = nn_time
    with netCDF4.Dataset(fc_folder + taus_fname) as ds:
        taus[i,:,:,0] = ds['tau'][:].data
    with netCDF4.Dataset(nn_folder + taus_fname) as ds:
        taus[i,:,:,1] = ds['tau'][:].data
    for j in range(2,6):
        bb_folder = bb_folders[j-2]
        bb_time = np.genfromtxt(bb_folder + time_fname)
        runtimes[i,j] = bb_time
        with netCDF4.Dataset(bb_folder + taus_fname) as ds:
            taus[i,:,:,j] = ds['tau'][:].data
    
    

plt.figure()
logbins = np.logspace(0, np.log10(800),20)
plt.hist(runtimes[:,0],bins=logbins,alpha=0.8,label='Full computation')
plt.hist(runtimes[:,1],bins=logbins,alpha=0.8,label='Nearest-neighbour interp.')
for j in range(2,6):
    plt.hist(runtimes[:,j],bins=logbins,alpha=0.8,label='Bounding-box interp., v_max = %f' % vmaxs[j])
plt.xscale('log')
plt.xlabel('runtime for cross-section computations (s)')
plt.title('Distribution of runtimes with different\ninterpolation schemes (40 atmospheres)')

plt.legend()
plt.show()

styles = ['k','r','b','b','b','b']
alphas = [1.0,1.0,1.0,0.6,0.4,0.2]
plt.figure(dpi=200)
for i in range(6):
    plt.plot(runtimes[:,i],styles[i],alpha=alphas[i])
plt.legend(('FC','NN','BB, v_max = %f'%vmaxs[0+2],'BB, v_max = %f'%vmaxs[1+2],'BB, v_max = %f'%vmaxs[2+2],'BB, v_max = %f'%vmaxs[3+2]),loc='upper left')
plt.ylabel('runtime for cross-section computations (s)')
plt.xlabel('index of atmosphere computed since the database initialization')
plt.savefig('runtimes.pdf')
plt.show()

relerr = True

for i in range(n_data):
    for j in range(n_layer):
        start_idx = (i * n_layer + j) * n_wn
        if relerr:
            diffs[start_idx : start_idx + n_wn,0] = (taus[i,j,:,0] - taus[i,j,:,1]) / taus[i,j,:,0]
            diffs[start_idx : start_idx + n_wn,1] = (taus[i,j,:,0] - taus[i,j,:,2]) / taus[i,j,:,0]
            ma[start_idx : start_idx + n_wn,0] = taus[i,j,:,0] > mask_check
        else:
            diffs[start_idx : start_idx + n_wn,0] = (taus[i,j,:,0] - taus[i,j,:,1])
            diffs[start_idx : start_idx + n_wn,1] = (taus[i,j,:,0] - taus[i,j,:,5])


tccon_filename = 'so20090516_20230530.public.qc.nc'
with netCDF4.Dataset(tccon_filename) as ds:
    tccon_altitude = ds['prior_altitude'][:].data

data_sel = 0
layer_sel = 0
#starti = 2520
#endi = 2530
starti = 0
endi = -1
#starti = 10190
#endi = 10210
#starti = 350
#endi = 410
plt.figure(dpi=200)
wl = np.linspace(1600,1700,n_wn)
leg = []
for l in [0,10,20,30,40,45][::-1]:
    plt.plot(wl,taus[data_sel,l,:,0],alpha=0.99)
    leg.append("%1.1f km" % tccon_altitude[l])
plt.legend(leg)
plt.ylabel('optical thickness (1/km)')
plt.xlabel('wavelength (nm)')
plt.savefig('example_taus.pdf')
plt.show()
for i in range(1,6):
    t = (taus[data_sel,layer_sel,starti:endi,0] - taus[data_sel,layer_sel,starti:endi,i])
    #plt.plot(taus[data_sel,layer_sel,starti:endi,0] - taus[data_sel,layer_sel,starti:endi,i])
    #plt.plot(t)
    plt.show()
    
    plt.plot(taus[data_sel,layer_sel,starti:endi,i])
    
#plt.plot(taus[data_sel,layer_sel,starti:endi,0])
plt.show()

plt.legend(('NN','BB_coarse','BB_fine','BB_extrafine','BB_ultrafine'))
plt.show()
for i in range(6):
    plt.plot(taus[data_sel,layer_sel,starti:endi,i])
plt.legend(('FC','NN','BB_coarse','BB_fine','BB_extrafine','BB_ultrafine'))
plt.show()
n_bins = 100
if not relerr:
    _logbins = np.logspace(np.log10(1e-12), np.log10(4),n_bins)
else:
    _logbins = np.logspace(np.log10(1e-5), np.log10(2),n_bins)
logbins = np.zeros((n_bins * 2))
logbins[:n_bins] = -_logbins[::-1]
logbins[n_bins:] = _logbins

plt.figure()
#plt.hist(diffs[np.abs(diffs[:,0]) > 1e-10,0],bins=logbins)

if relerr:
    a = plt.hist(diffs[ma[:,0],0],bins=_logbins,alpha=0.7,label='(FC - BB)/FC')
    b = plt.hist(-diffs[ma[:,0],0],bins=_logbins,alpha=0.7,label='(BB - FC)/FC')
    plt.xscale('log')
else:
    a = plt.hist(diffs[:,0],bins=_logbins,alpha=0.7,label='FC - BB')
    b = plt.hist(-diffs[:,0],bins=_logbins,alpha=0.7,label='BB - FC')
    plt.xscale('log')
error_prop = 1 - (np.sum(a[0]) + np.sum(b[0])) / diffs[:,0].size
plt.title('Tau interpolation errors using bounding-box algorithm\nErrors less than 10^-12: %1.2f percent (not shown)' % (error_prop * 100))
plt.legend()
plt.show()

plt.figure()
#plt.hist(diffs[np.abs(diffs[:,0]) > 1e-10,0],bins=logbins)
a = plt.hist(diffs[:,1],bins=_logbins,alpha=0.7,label='FC - NN')
b = plt.hist(-diffs[:,1],bins=_logbins,alpha=0.7,label='NN - FC')
plt.xscale('log')
error_prop = 1 - (np.sum(a[0]) + np.sum(b[0])) / diffs[:,0].size
plt.title('Tau interpolation errors using nearest-neighbour algorithm\nErrors less than 10^-12: %1.2f percent (not shown)' % (error_prop * 100))
plt.legend()
plt.show()



# yhdistä noi, ja sitten vielä jokin jakauma niistä atmosfääriparametreista subplotteina
# esimerkkispektrejäkin!
# idea: voisiko tehdä 2-d histogrammin että x-akselilla oisi spektri ja y oisi error?
false = False
if false:
    for i in range(n_data):
        for j in range(n_layer):
            plt.figure()
            plt.plot(taus[i,j,:,0] - taus[i,j,:,2])
            plt.plot(taus[i,j,:,0] - taus[i,j,:,3])
            plt.plot(taus[i,j,:,0] - taus[i,j,:,4])
            plt.plot(taus[i,j,:,0] - taus[i,j,:,5])
            plt.show()

    
thicc = np.diff(tccon_altitude)
for interpolator_c in range(1,6):
    #interpolator_c = 5
    l = []
    x = []
    
    for sel_p in range(40):
        T_real = np.exp(-taus[sel_p,:,:,0].T @ thicc)
        for interpolator in range(1,6):
            T_int = np.exp(-taus[sel_p,:,:,interpolator].T @ thicc)
            #plt.plot((T_real-T_int)/T_real,alpha=0.5)
            nz_ma = T_real > 1e-6
            relerrs = (T_real[nz_ma]-T_int[nz_ma])/T_real[nz_ma]
            relerrs = (T_real-T_int)/T_real
            if interpolator_c == interpolator:
                l.append(relerrs*100)
                x.append(sel_p)
            #print(np.mean(relerrs),np.std(relerrs),np.max(relerrs),np.min(relerrs))
        #plt.show()
    maxes = []
    mins = []
    std = []
    means = []
    for i in range(40):
        maxes.append(max(l[i]))
        mins.append(min(l[i]))
        std.append(np.std(l[i]))
        means.append(np.mean(l[i]))
    std_env = 3
    plt.figure(dpi=200)
    #for i in range(40):    
    plt.plot(x,means,'b',label='mean rel. error')
    plt.plot(x,np.array(means) + std_env * np.array(std),'r--',label='%d-sigma envelope' %std_env)
    plt.plot(x,np.array(means) - std_env * np.array(std),'r--')
    plt.plot(x,mins,'k:',alpha=0.5,label='outlier envelope')
    plt.plot(x,maxes,'k:',alpha=0.5)
    plt.xlabel('index of atmosphere computed since the database initialization')
    plt.ylabel('relative error (%)')
    titlestr = 'Nearest-neighbour, v_max = %f' if interpolator_c == 1 else 'Bounding-box, v_max = %f'
    plt.title(titlestr % vmaxs[interpolator_c])
    ylims = [-3,4] if interpolator_c == 1 else [-17,52]
    plt.ylim(ylims)
    plt.legend()
    plt.savefig('interp_%d.pdf' %interpolator_c)
    plt.show()
# vastaavanlaiset virhe-enveloopit kullekin eri ilmakehälle? kuten siis se aikakäppyrä
# kokoa transmittanssista jakaumat kullekin ilmakehälle ja niistä virheet siis!
