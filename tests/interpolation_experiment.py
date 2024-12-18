#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:17:15 2024

@author: mikkonea
"""

import netCDF4
import numpy as np

# file downloaded from https://data.caltech.edu/records/fnpcw-myc91
tccon_filename = 'so20090516_20230530.public.qc.nc'
settings_file = './config.yaml'

"""
REMEMBER to delete all tables before changing the mode!

"""
import sys

interp_mode = sys.argv[1]
bb_mode = sys.argv[2]
run_mode = 'interpolate'
#run_mode = 'full_compute'
#interp_mode = 'bb'
#bb_mode = 'coarse'
#bb_mode = 'fine'
#bb_mode = 'extrafine'
#bb_mode = 'ultrafine'
#interp_mode = 'nn'

def change_settings(setting,value):
    """
    Changes hapi2libis settings
    Both setting and value need to be in string format.
    """
    row_idx = "" # This needs to be preset so it exists in this scope
    # and preferably it needs to be non-index so an error gets raised if
    # such setting isn't find (just an extra countermeasure for the for-else)
    extra_indent = 0
    with open(settings_file,'r') as f:
        lines = f.readlines()
        for l_idx, l in enumerate(lines):
            ls = l.strip()
            lsl = ls.split(':')
            if lsl[0].strip() == setting:
                row_idx = l_idx
                extra_indent = len(l) - len(l.lstrip())
                break
        else:
            raise ValueError('Could not find setting "%s" in the settings file!' % setting)
    with open(settings_file,'w') as f:
        lines[row_idx] = "%s%s : %s\n" % (extra_indent * " ",setting,value)
        f.writelines(lines)


if run_mode == 'interpolate':
    change_settings('use_xsec_db','True')
else:
    change_settings('use_xsec_db','False')

if interp_mode == 'bb':
    #bb_str = 'ultrafine'
    bb_folder = './uvspec_interpolated_bb_%s/' % bb_mode
    change_settings('interpolation_method', "'bounding-box'")
    if bb_mode == 'coarse':
        change_settings('interpolation_error_tolerance', "0.001") #coarse
    elif bb_mode == 'fine':    
        change_settings('interpolation_error_tolerance', "0.0001") #fine
    elif bb_mode == 'extrafine':    
        change_settings('interpolation_error_tolerance', "0.00001") #extrafine
    elif bb_mode == 'ultrafine':
        change_settings('interpolation_error_tolerance', "0.000001") # ultrafine
    else:
        raise
elif interp_mode == 'nn':
    change_settings('interpolation_method', "'nearest-neighbour'")
    change_settings('interpolation_error_tolerance', "0.1")

change_settings('wl_range_nm','[1600,1700]')
change_settings("selected_gases","['H2O','CO2','CH4']")
#change_settings("selected_gases","['O2','H2O','CO2']")


with netCDF4.Dataset(tccon_filename) as ds:
    tccon_o2 = ds['prior_o2'][:].data
    tccon_co2 = ds['prior_co2'][:].data
    #tccon_o3 = ds['prior_o3'][:].data
    #tccon_no2 = ds['prior_no2'][:].data
    tccon_h2o = ds['prior_h2o'][:].data
    tccon_ch4 = ds['prior_ch4'][:].data
    tccon_time = ds['prior_time'][:].data
    tccon_day = ds['day'][:].data
    tccon_year = ds['year'][:].data
    tccon_altitude = ds['prior_altitude'][:].data
    tccon_temperature = ds['prior_temperature'][:].data
    tccon_pressure = ds['prior_pressure'][:].data
    tccon_density = ds['prior_density'][:].data

ma_tccon_year = tccon_year == 2023
ma_tccon_day = np.logical_and(tccon_day >= 60, tccon_day <= 90)
ma_tccon = np.logical_and(ma_tccon_year,ma_tccon_day)

sel_idxs = np.where(ma_tccon)[0]

# file format for custom atmos
# altitude (km)     p(mb)      T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)     h2o(cm-3)     co2(cm-3)     no2(cm-3)

atmos_folder = './atmos_files/'
if run_mode == 'interpolate':
    if interp_mode == 'bb':
        uvspec_folder = bb_folder
    elif interp_mode == 'nn':
        uvspec_folder = './uvspec_interpolated_nn/'
else:
    uvspec_folder = './uvspec_full_compute/'

tot_atmos = 40
start_idx = 198589
# until the day 102 of year 2023

atmos_name_template = 'sod_atmos_%d.dat'
atmos_names = []
i_atmos = 0
prescheck = tccon_pressure[0,::-1]

while len(atmos_names) < tot_atmos:
    i_atmos += 1
    if np.linalg.norm(prescheck - tccon_pressure[start_idx + i_atmos,::-1]) > 0.0:
        #a new atmos was found! let's save it!
        idx_time = start_idx + i_atmos
        prescheck = tccon_pressure[idx_time,::-1]
    else:
        continue
    atmos = np.zeros((tccon_density.shape[1],10))
    atmos[:,0] = tccon_altitude[::-1]
    atmos[:,1] = tccon_pressure[idx_time,::-1] * 1013.25
    atmos[:,2] = tccon_temperature[idx_time,::-1]
    atmos[:,3] = tccon_density[idx_time,::-1]
    atmos[:,4] = tccon_density[idx_time,::-1] * 0.0 #o3
    atmos[:,5] = tccon_o2[idx_time,::-1] * tccon_density[idx_time,::-1] * 0.0
    atmos[:,6] = tccon_h2o[idx_time,::-1] * tccon_density[idx_time,::-1]
    atmos[:,7] = tccon_co2[idx_time,::-1] * tccon_density[idx_time,::-1] * 1e-6
    atmos[:,8] = tccon_density[idx_time,::-1] * 0.0 # no2
    atmos[:,9] = tccon_ch4[idx_time,::-1] * tccon_density[idx_time,::-1] * 1e-9
    atmos_name = atmos_name_template % idx_time
    np.savetxt(atmos_folder + atmos_name,atmos,delimiter='   ')
    atmos_names.append(atmos_name)
    
import os

for atmos_idx in range(40):
    change_settings("'filepath'","'%s%s'" % (atmos_folder,atmos_names[atmos_idx]))
    change_settings("name_suffix","'%d'" % atmos_idx)
    change_settings("xsec_db_folder","'csdb_%s'" % bb_mode)
    os.system("python hapi2libis.py")
    uvspec_fname = "uvspec_custom_%d.nc" % atmos_idx
    os.system("mv %s %s%s" % (uvspec_fname,uvspec_folder,uvspec_fname))
    time_fname = "time_%d.dat" % atmos_idx
    os.system("mv %s %s%s" % (time_fname,uvspec_folder,time_fname))
    
