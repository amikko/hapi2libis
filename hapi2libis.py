#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# HAPI2LIBIS v1.0
# Authors: Antti Kukkurainen1 & Antti Mikkonen1
# 1: Finnish Meteorological Institute
# Correspondence: antti.kukkurainen@fmi.fi


import hapi
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import numpy as np
from netCDF4 import Dataset
import os
import netCDF4
import time
import yaml
import sys
    
try:
    with open(sys.argv[1], 'r') as settings_file:
        settings = yaml.safe_load(settings_file)  
    print(f"Settings loaded from file: {sys.argv[1]}\n")

except(IndexError):
    print('\nNo settings file given for HAPI2LIBIS!')
    print("Trying to load default settings file: config.yaml")
    
    with open('config.yaml', 'r') as settings_file:
        settings = yaml.safe_load(settings_file)  
    
    print("Settings loaded!\n")
        
# Desired wavelength range in nanometers
wl_range_nm = settings['wl_range_nm']

# HAPI resolution in wavenumbers
nu_resolution = settings['nu_resolution']

# Write molecular absorption netcdf file for libradtran
create_mol_abs_netcdf_file = settings['create_mol_abs_netcdf_file']

# Name suffix for the created molecular absorption file
name_suffix = settings['name_suffix']

# List of gases to compute
selected_gases = settings['selected_gases']

# Force edges to range
boundary_flag = settings['boundary_flag']

# Compute at level or layer midpoint
slab_midpoint_flag = settings['interpolate_to_layer_midpoint']

# Return maximum range for non-reflectivity calculations.
# Requires full kurucs irradiance file from libRadtran!
check_wavelength_limits = settings['check_wavelength_limits']

# Load profiles for CH4, N2O, CO and N2 from auxiliary US-standard files
load_aux_gases_from_us_standard = settings['load_aux_gases_from_us_standard']

# Choose atmosphere file
atmosphere_id = settings['atmosphere_id']

# Custom atmosphere loading requires a dict with gases column number
# Check variable "atm_col_id_dict" in for example
# Columns z (km), p (mb), T (K), air (cm-3) are minimum requirement
custom_atm_dict = settings['custom_atm_dict']

# Define if gas cross-section data should come from libRadtran tables.
# The first one is the default. 
crs_table_gases = settings['crs_table_gases']

# Location of main libRadtran folder
libradtranpath = settings['libradtranpath']

# Broadening function choice
function_choice = settings['function_choice']

# Folder to save HITRAN data
hapi_dl_dir = settings['hapi_dl_dir']

# Irregular interpolation options
use_xsec_db = settings['use_xsec_db']
xsec_db_folder = settings['xsec_db_folder']
cl_interp_p = settings['cl_interp_p'] # set this to 0 to force the computation of the cross-sections
interpolation_method = settings['interpolation_method']
# The interpolation_error_tolerance is the maximum volume of the box allowed
# for interpolation, or the maximum distance from particular point
interpolation_error_tolerance = settings['interpolation_error_tolerance']

###-------------------------------------------------------------------------###
###-------------------------------------------------------------------------###
###-------------------------------------------------------------------------###

atmosphere_dict = {1:'afglms', # midlatitude summer
                   2:'afglmw', # midlatitude winter
                   3:'afglss', # subarctic summer
                   4:'afglsw', # subarctic winter
                   5:'afglt',  # tropical
                   6:'afglus', # US-standard
                   'custom': custom_atm_dict
                   }

def load_default_atmosphere():
    
    atm_file = '{}/data/atmmod/{}.dat'.format(libradtranpath, atmosphere_dict[atmosphere_id])
    # header = ['z(km)', 'p(mb)', 'T(K)', 'air(cm-3)', 'o3(cm-3)', 'o2(cm-3)', 'h2o(cm-3)', 'co2(cm-3)', 'no2(cm-3)']
    # atm_df = pd.read_csv(atm_file, sep='\s+', comment='#', names=header)
    atm_arr = np.loadtxt(atm_file, comments='#')
    
    if load_aux_gases_from_us_standard:
        afglus_files = [f'{libradtranpath}/data/atmmod/afglus_ch4_vmr.dat',
                        f'{libradtranpath}/data/atmmod/afglus_n2o_vmr.dat',
                        f'{libradtranpath}/data/atmmod/afglus_co_vmr.dat',
                        f'{libradtranpath}/data/atmmod/afglus_n2_vmr.dat']
    
        for aux_file_idx in afglus_files:
            aux_gas = np.loadtxt(aux_file_idx, comments='#')
            assert np.isclose(aux_gas[:, 0], atm_arr[:, 0]).all()
            # Unit conversion
            aux_gas[:, 1] *= atm_arr[:, 3]
            atm_arr = np.hstack((atm_arr, aux_gas[:, 1][:, None]))
    return atm_arr


# Load atmosphere
if atmosphere_id in range(1, 7):
    atm_arr = load_default_atmosphere()
    # Numbers correspond to columns in atm file
    atm_col_id_dict = {'altitude': 0, 'pressure': 1, 'temperature': 2, 'air': 3,
                       'O3': 4, 'O2': 5, 'H2O': 6, 'CO2': 7, 'NO2': 8, 
                       'CH4': 9, 'N2O': 10, 'CO': 11,'N2': 12}
elif atmosphere_id == 'custom':
    atm_arr = np.loadtxt(atmosphere_dict['custom']['filepath'], comments='#')
    atm_col_id_dict = atmosphere_dict['custom']['col_ids']
else:
    raise Exception(f"INCORRECT 'atmosphere_id': {atmosphere_id}")


function_listing = {'Lorentz' : hapi.absorptionCoefficient_Lorentz,
                    'Doppler' : hapi.absorptionCoefficient_Doppler,
                    'Voigt'   : hapi.absorptionCoefficient_Voigt,
                    'HT'      : hapi.absorptionCoefficient_HT,
                    'SDVoigt' : hapi.absorptionCoefficient_SDVoigt}
broadening_func = function_listing[function_choice]

# Initialize HAPI
hapi.db_begin(hapi_dl_dir)

# Convert to wavenumbers for HAPI
nu_max = 1 / (wl_range_nm[0] * 1e-7)
nu_min = 1 / (wl_range_nm[1] * 1e-7)

global_gas_ids = {
    'H2O' : [1,2,3,4,5,6,129],
    'CO2' : [7,8,9,10,11,12,13,14,121,15,120,122],
    'O3' : [16,17,18,19,20],
    'N2O' : [21,22,23,24,25],
    'CO' : [26,27,28,29,30,31],
    'CH4' : [32,33,34,35],
    'O2' : [36,37,38],
    'NO' : [39,40,41],
    'SO2' : [42,43,137,138],
    'NO2' : [44,130],
    'NH3' : [45,46],
    'HNO3' : [47,117],
    'OH' : [48,49,50],
    'HF' : [51,110],
    'HCl' : [52,53,107,108],
    'HBr' : [54,55,111,112],
    'HI' : [56,113],
    'ClO' : [57,58],
    'OCS' : [59,60,61,62,63,135],
    'H2CO' : [64,65,66],
    'HOCl' : [67,68],
    'N2' : [69,118],
    'HCN' : [70,71,72],
    'CH3Cl' : [73,74],
    'H2O2' : [75],
    'C2H2' : [76,77,105],
    'C2H6' : [78,106],
    'PH3' : [79],
    'COF2' : [80,119],
    'SF6' : [126],
    'H2S' : [81,82,83],
    'HCOOH' : [84],
    'HO2' : [85],
    'O' : [86],
    'ClONO2' : [127,128],
    'NO+' : [87],
    'HOBr' : [88,89],
    'C2H4' : [90,91],
    'CH3OH' : [92],
    'CH3Br' : [93,94],
    'CH3CN' : [95],
    'CF4' : [96],
    'C4H2' : [116],
    'HC3N' : [109],
    'H2' : [103,115],
    'CS' : [97,98,99,100],
    'SO3' : [114],
    'C2N2' : [123],
    'COCl2' : [124,125],
    'SO' : [146,147,148],
    'CH3F' : [144],
    'GeH4' : [139,140,141,142,143],
    'CS2' : [131,132,133,134],
    'CH3I' : [145],
    'NF3' : [136]}

gas_dict = {}

for g in selected_gases:
    gas_dict[g] = global_gas_ids[g]


#1) Name of local table
#2) molecule and isotope HITRAN numbers (M & I) or global ID if fetch_by_ids (iso_ID)
#3-4) wavenumber range (nu_min-nu_max)
# hapi.fetch('O3', 3, 1, nu_min, nu_max)
# hapi.fetch('O2', 7, 1, nu_min, nu_max)
# hapi.fetch('H2O', 1, 1, nu_min, nu_max)
# hapi.fetch('CO2', 2, 1, nu_min, nu_max)
# hapi.fetch('NO2', 10, 1, nu_min, nu_max)

# fetch_by_ids is easier to use when we want multiple isotopes

# --------------------------------
# The cross-section database stuff
# --------------------------------


def folder_csdb():
    return xsec_db_folder + '/'

def if_folder_not_exist_then_create(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

def setup_file(subst,wn,filename):
    """
    This function sets up a netCDF4 datafile for cross-sections of substance
    subst for wavenumbers wn.

    The spectra are static, which means that in your use case you need to
    store the reasonable maximum bandwidth.
    """

    if_folder_not_exist_then_create(folder_csdb())

    if os.path.isfile(filename):
        return
    print("Creating new file '%s'..." % filename)
    with netCDF4.Dataset(filename, 'w', format="NETCDF4") as ds:
    
        ds.description = "Cross-sections for substance '%s'." % (subst)
        ds.history = "Created %s." % time.ctime(time.time())
    
        len_wn = wn.size
    
        #the dimensions are set to a variable for clarity of the nc data structure
        dim_tpps_idx = ds.createDimension('TPPs_index', 0)
        dim_wl = ds.createDimension('wavenumber', len_wn)
        dim_cubic = ds.createDimension('dim_scale',3)
    
        var_xsec = ds.createVariable('cross_section', 'f4' , ('TPPs_index','wavenumber',))
        var_T = ds.createVariable('temperature', 'f4' , ('TPPs_index',))
        var_p = ds.createVariable('pressure', 'f4' , ('TPPs_index',))
        var_ps = ds.createVariable('partial_pressure', 'f4' , ('TPPs_index',))
        var_wn = ds.createVariable('wavenumber', 'f4' , ('wavenumber',))
        var_dim = ds.createVariable('dimension_scaling', 'f4', ('dim_scale',))
    
        var_wn[:] = wn
        var_dim[:] = np.array([1,1,1])
    
        var_xsec.units = 'cm^2'
        var_T.units = 'K'
        var_p.units = 'atm'
        var_ps.units = 'atm'
        var_wn.units = '1/cm'

def filenamefun(subst,wn_min,wn_max,wn_reso,function_choice):
    return folder_csdb() + '%s_%f_%f_%f_%s.nc' % (subst,wn_min,wn_max,wn_reso,function_choice)
def input_xsec_data(subst,T,P,Ps,wn_min,wn_max,wn_reso,wn,xsec):
    """
    Appends the data xsec(TPPs,wn) into the file basename_subst.nc.
    """
    filename = filenamefun(subst,wn_min,wn_max,wn_reso,function_choice)
    if not os.path.isfile(filename):
        print("File '%s' not found! Setting up..." % filename)
        setup_file(subst,wn,filename)
    with netCDF4.Dataset(filename, 'a', format="NETCDF4") as ds:
        assert np.isclose(wn,ds['wavenumber']).all(), ("The wavenumbers 'wn' " +
        "aren't all equal with the ones in the file %s!" % filename)
        TPPs_len = T.size
        ds_len = ds['cross_section'].shape[0]
    
        start_idx = ds_len
        end_idx = start_idx + TPPs_len
        ds['temperature'][start_idx : end_idx] = T
        ds['pressure'][start_idx : end_idx] = P
        ds['partial_pressure'][start_idx : end_idx] = Ps
        ds['cross_section'][start_idx : end_idx, :] = xsec
        weights = np.array([(np.max(ds['temperature'][:]) - np.min(ds['temperature'][:])) ** -2,
                              (np.max(ds['pressure'][:]) - np.min(ds['pressure'][:])) ** -2,
                              (np.max(ds['partial_pressure'][:]) - np.min(ds['partial_pressure'][:])) ** -2])
        ds['dimension_scaling'][:] = weights

def load_xsec_data(subst,wn_range,wn_step):
    """
    Loads all xsec-data from the file basename_subst.nc
    """
    filename = filenamefun(subst,wn_range[0],wn_range[1],wn_step,function_choice)
    if not os.path.isfile(filename):
        print("File '%s' not found! Computing all the cross-sections..." % filename)
        return (None, None, None, None, None, None)
    with netCDF4.Dataset(filename, 'r', format="NETCDF4") as ds:

        T = ds['temperature'][:].data
        P = ds['pressure'][:].data
        Ps = ds['partial_pressure'][:].data
        wn = ds['wavenumber'][:].data
        xsec = ds['cross_section'][:].data
        W = ds['dimension_scaling'][:].data

    return (T,P,Ps,wn,xsec,W)

# -----------------------------------------
# The interpolation scheme for the database
# -----------------------------------------

def is_inside_box(x,a,b):
    """
    This function checks if x is within an n-box defined by a and b as its
    vertices.
    """
    ab = np.array([a,b])
    min_corner = ab.min(0)
    max_corner = ab.max(0)
    return np.all(x <= max_corner) and np.all(min_corner <= x)

def get_distances(x, p, W):
    n = p.shape[1]
    distance_weights = W
    W_ = np.diag(distance_weights)
    dists = np.inf * np.ones(n)
    for i in range(0,n):
        dists[i] = (p[:,i] - x) @ W_ @ (p[:,i] - x)
    return dists

def find_closest_element(x, p, W):
    dists = get_distances(x, p, W)
    return np.argmin(dists)

def box_volume(a,b,W):
    return np.prod(np.multiply(np.absolute(a - b),W))

def find_smallest_box_bf(x, p, W):
    """
    This function finds the smallest enclosing box for the parameters by
    comparing all of them.

    Complexity: O(n^2)
    """
    n = p.shape[1]
    boxes = []
    measures = []
    for i in range(0,n):
        for j in range(0,i):
            if is_inside_box(x,p[:,i],p[:,j]):
                boxes.append((i,j))
                measures.append(box_volume(p[:,i],p[:,j]),W)
    if not boxes:
        return(-1,find_closest_element(x,p,W))
    else:
        return(boxes[np.argmin(measures)])

def find_smallest_box(x, p, lim, W):
    """
    This function finds the smallest enclosing box for the parameters.

    Find the closest point and test other points if they enclose x. If they do,
    say that the box is smallest. If not, then select the second-closest point
    and try again. If a box isn't found, then the value is calculated.

    Complexity: O(n)
    
    TODO: in degenerate cases, such as when the box is zero-measure, then there should be
    a more thorough edge case handling.
    """
    n = p.shape[1]
    if lim == -1:
        #in this case we interpolate from all the values in the database.
        lim = n
    dists = get_distances(x, p, W)
    dist_ord = dists.argsort()
    n_box = min(lim,n)
    boxes = []
    measures = []
    #box_idxs_vol = np.zeros((n_box,3))
    for l in range(n_box):
        for i in range(l + 1,n):
            if is_inside_box(x,p[:,dist_ord[l]],p[:,dist_ord[i]]):
                boxes.append((dist_ord[l],dist_ord[i]))
                measures.append(box_volume(p[:,dist_ord[l]],p[:,dist_ord[i]],W))
    if not boxes:
        return(-1,dist_ord[0])
    else:
        return boxes[np.argmin(measures)]

def interpolate_bounding_box(x,a,b,fa,fb,W):
    W_ = np.diag(W)
    da = (a - x) @ W_ @ (a - x)
    db = (b - x) @ W_ @ (b - x)
    t = da / (da + db)
    return fa * (1 - t) + fb * t

def interpolate_nearest_neighbour(x,a,b,fa,fb,W):
    return fa 

if interpolation_method == 'bounding-box':
    interpolate = interpolate_bounding_box
elif interpolation_method == 'nearest-neighbour':
    interpolate = interpolate_nearest_neighbour

def compute_dimension_scaling(subst,wn_min,wn_max,wn_reso):
    # TODO: Should other W-elements be fitted too? It could boost the interpolation?
    # Antti M. (7.6.2024): The experiments with this show that in this formulation
    # at least, there is no guarantee of unique minimum for W with particular point cloud.
    # This approach should be set onto the backburner and instead of use the weights W
    # by scaling each of the dimensions with max(dim) - min(dim)
    filename = filenamefun(subst,wn_min,wn_max,wn_reso,function_choice)
    Tvals,Pvals,Psvals,wn,xsec,W = load_xsec_data(subst,[wn_min,wn_max],wn_reso)
    max_vals = [np.max(Tvals), np.max(Pvals), np.max(Psvals)]
    Tvals = Tvals / max_vals[0]
    Pvals = Pvals / max_vals[1]
    Psvals = Psvals / max_vals[2]
    def find_interpolation_residue(W_):
        residy = 0
        for i in range(Tvals.size):
            ma = np.zeros((Tvals.size,),dtype=np.bool_)
            ma[i] = True
            ma = np.logical_not(ma)
            TPPs = np.array([Tvals[ma].ravel(), Pvals[ma].ravel(), Psvals[ma].ravel()])
            x = np.array([Tvals[i],Pvals[i],Psvals[i]])
            (p_i,p_j) = find_smallest_box(x,TPPs,Tvals.size,W_)
            if p_i != -1:
                cs_interp = interpolate(x,TPPs[:,p_i],TPPs[:,p_j],xsec[p_i][:],xsec[p_j][:],W_)
                residy += np.sum(np.abs(cs_interp - xsec[i][:]))
        residy += np.linalg.norm(W_)
        return (residy)
    def jacobian(W_):
        # This is defined because some minimization methods cannot handle the 
        # dimenions properly. Nelder-Mead does not require this.
        diff_prop = 0.1
        mul = 1 + diff_prop
        init = find_interpolation_residue(W_)
        diff1 = (find_interpolation_residue(W_ * np.array([mul,1.0,1.0])) - init)
        diff2 = (find_interpolation_residue(W_ * np.array([1.0,mul,1.0])) - init)
        diff3 = (find_interpolation_residue(W_ * np.array([1.0,1.0,mul])) - init)
        return np.array([diff1,diff2,diff3])
    #minimize(find_interpolation_residue, W)
    res = minimize(find_interpolation_residue, np.array([1.5,1.0,1.5]),method='Nelder-Mead',tol=1e-5,options={'maxiter':250})
    W = np.array([res.x[0],res.x[1],res.x[2]])
    print(W)
    #store W into the data
    return res

#res = compute_dimension_scaling('CO2',nu_min,nu_max,nu_resolution)

def database_interpolation(gas,wn_min,wn_max,wn_step,T,P,Ps):

    did_we_interp = []
    interp_box_indices = []
    interp_box_size = []

    Tvals,Pvals,Psvals,wn,xsec,W = load_xsec_data(gas,[wn_min,wn_max],wn_step)
    
    fresh_csdb = (type(Tvals) == type(None)) #couldn't find the nc-file

    comp_xsec_idxs = []
    if fresh_csdb:
        for alt_idx in range(0,T.size):
            comp_xsec_idxs.append(alt_idx)
        wnAmt = np.array(np.arange(wn_min,wn_max,wn_step)).size
        cs = np.zeros((T.size,wnAmt))
        return wn, None
    else:
        TPPs = np.array([Tvals.ravel(), Pvals.ravel(), Psvals.ravel()])
        cs = np.zeros((T.size,wn.size))
        for alt_idx in range(0,T.size):
            x = np.array([T[alt_idx],P[alt_idx],Ps[alt_idx]])
            if TPPs.size > 0:
                for tpps_idx in range(TPPs.shape[1]):
                    if np.all(np.isclose(x,TPPs[:,tpps_idx])):
                        #an almost exact same data point is found
                        #let's use that one straight away
                        i = tpps_idx
                        j = tpps_idx
                        break
                else:
                    if np.prod(W) == np.inf or np.isnan(np.prod(W)):
                        # this is the case when there is no data in the csdb
                        i = -1
                    elif interpolation_method == 'bounding-box':
                        (i,j) = find_smallest_box(x,TPPs,cl_interp_p,W)
                        box_space_diagonal_ = TPPs[:,i] - TPPs[:,j]
                        box_space_diagonal = box_space_diagonal_ * np.sqrt(W)
                        box_vol = np.prod(box_space_diagonal)
                        if box_vol > interpolation_error_tolerance:
                            i = -1
                    elif interpolation_method == 'nearest-neighbour':
                        dists = get_distances(x, TPPs, W)
                        dist_ord = dists.argsort()
                        i = dist_ord[0]
                        j = dist_ord[0]
                        if np.linalg.norm(x - TPPs[:,i]) > interpolation_error_tolerance:
                            i = -1
                    else:
                        raise("No interpolation method '%s'" % interpolation_method)
            else: #if there's no data yet
                i = -1

            if i == -1:
                #no box found -> compute new point
                comp_xsec_idxs.append(alt_idx)
            else:
                #box found -> interpolate
                cs[alt_idx,:] = interpolate(x,TPPs[:,i],TPPs[:,j],
                xsec[i][:],xsec[j][:],W)
            if i == -1:
                did_we_interp.append(False)
                interp_box_indices.append([-1,-1])
                interp_box_size.append(-1)
            else:
                did_we_interp.append(True)
                interp_box_indices.append([i,j])
                interp_box_size.append(box_volume(TPPs[:,i],TPPs[:,j], W))

    if did_we_interp[0]:
        return wn, cs
    else:
        return wn, None


# Fetch gases from HITRAN
fetched_gases = []
def fetch_lines(gas_fetch, nu_min, nu_max):
    # Try to load gases and ignore a gas if HITRAN returns nothing
    try:
        hapi.fetch_by_ids(gas_fetch, gas_dict[gas_fetch], nu_min, nu_max)
        return True
    except Exception:
        return False

for gas in gas_dict:
    fetched = fetch_lines(gas, nu_min, nu_max)
    if fetched:
        fetched_gases.append(gas)
        

lookup_table_gases = []

def ozone_cross_section(wavelength, temperature, data):
    
    if (np.min(wavelength) < data[0, 0]) or (np.max(wavelength) > data[-1, 0]):
        print("Warning! Requested wavelength is outside the table wavelength range!")
    
    C0_interp = np.interp(wavelength, data[:, 0], data[:, 1], left=0, right=0)
    C1_interp = np.interp(wavelength, data[:, 0], data[:, 2], left=0, right=0)
    C2_interp = np.interp(wavelength, data[:, 0], data[:, 3], left=0, right=0)
    
    T = temperature
    
    if crs_table_gases['O3']['id'] in ['molina', 'bass_and_paur', 'daumount']:
        T0 = 273.13
        sigma = (C0_interp + C1_interp*(T-T0) + C2_interp*(T-T0)**2) * 1.E-20
    elif crs_table_gases['O3']['id'] in ['bogumil', 'bogumil_exp']:
        T0 = 273.15
        if crs_table_gases['O3']['id'] == 'bogumil_exp':
            # Alternative equation from Bogumil et al. 2003
            sigma = (C0_interp * np.exp( - C1_interp*T + C2_interp/T)) * 1.E-20
        else:
            sigma = (C0_interp + C1_interp*(T-T0) + C2_interp*(T-T0)**2) * 1.E-20
    else:
        raise Exception('Oops! Unknown table id!')
    
    return sigma

def nitrogen_oxide_cross_section(wavelength, temperature, data):
    if (np.min(wavelength) < data[0, 0]) or (np.max(wavelength) > data[-1, 0]):
        print("Warning! Requested wavelength is outside the table wavelength range!")
    
    if crs_table_gases['NO2']['id'] in ['burrows', 'schneider']:
        sigma = np.interp(wavelength, data[:, 0], data[:, 1], left=0, right=0)
    elif crs_table_gases['NO2']['id'] in ['bogumil', 'bogumil_exp']:
        T0 = 273.15
        T = temperature
        C0_interp = np.interp(wavelength, data[:, 0], data[:, 1], left=0, right=0)
        C1_interp = np.interp(wavelength, data[:, 0], data[:, 2], left=0, right=0)
        C2_interp = np.interp(wavelength, data[:, 0], data[:, 3], left=0, right=0)
        if crs_table_gases['O3']['id'] == 'bogumil_exp':
            # Alternative equation from Bogumil et al. 2003
            sigma = (C0_interp * np.exp( - C1_interp*T + C2_interp/T)) * 1.E-20
        else:
            sigma = (C0_interp + C1_interp*(T-T0) + C2_interp*(T-T0)**2) * 1.E-20
    else:
        raise Exception('Oops! Unknown table id!')
    return sigma

def oxygen_dimer(wavelength, data):
    if (np.min(wavelength) < data[0, 0]) or (np.max(wavelength) > data[-1, 0]):
        print("Warning! Requested wavelength is outside the table wavelength range!")
    sigma = np.interp(wavelength, data[:, 0], data[:, 1], left=0, right=0)
    return sigma


# Check that table gases are not downloaded by HAPI
for crs_table_gas_id in crs_table_gases:
    if crs_table_gases[crs_table_gas_id]['use']:
        if crs_table_gas_id in fetched_gases:
            print(f"\nWARNING! HAPI has downloaded {crs_table_gas_id} lines." * 10)
            print("Beware of possible overlap with table values!")
        lookup_table_gases.append(crs_table_gas_id)

if crs_table_gases['O3']['use']:
    table_O3_paths = {'molina': f'{libradtranpath}/data/crs/crs_o3_mol_cf.dat',
                      'bass_and_paur': f'{libradtranpath}/data/crs/crs_o3_pab_cf.dat', 
                      'daumount': f'{libradtranpath}/data/crs/crs_o3_dau_cf.dat', 
                      'bogumil': f'{libradtranpath}/data/crs/crs_O3_UBremen_cf.dat'}
    table_O3 = np.loadtxt(table_O3_paths[crs_table_gases['O3']['id']], comments='#')
    
if crs_table_gases['NO2']['use']:
    table_NO2_paths = {'burrows': f'{libradtranpath}/data/crs/crs_no2_gom.dat',
                       'bogumil': f'{libradtranpath}/data/crs/crs_NO2_UBremen_cf.dat', 
                       'schneider': f'{libradtranpath}/data/crs/crs_no2_012.dat'}
            
    table_NO2 = np.loadtxt(table_NO2_paths[crs_table_gases['NO2']['id']], comments='#')
    
if crs_table_gases['O2-O2']['use']:    
    table_O2_O2_paths = {'greenblatt': f'{libradtranpath}/data/crs/crs_o4_greenblatt.dat'}
    
    table_O2_O2 = np.loadtxt(table_O2_O2_paths[crs_table_gases['O2-O2']['id']])

if __name__ == '__main__':
    import time
    # this is to time the computations
    # the lines are downloaded every time, so we do not want to include
    # that into the script timing.
    
    # Generate interpolators for atm
    altitude = atm_arr[:, atm_col_id_dict['altitude']]
    pressure = atm_arr[:, atm_col_id_dict['pressure']]
    temperature = atm_arr[:, atm_col_id_dict['temperature']]
    
    pressure_int = interp1d(altitude, pressure)
    temperature_int = interp1d(altitude, temperature)
    
    # Generate interpolators for gases
    gas_int_dict = {}
    
    gas_int_dict.update({'air':interp1d(altitude, atm_arr[:, atm_col_id_dict['air']])})
    for gas_int in fetched_gases:
        gas_int_dict.update({gas_int:interp1d(altitude, atm_arr[:, atm_col_id_dict[gas_int]])})
        
    for gas_int in lookup_table_gases:
        if gas_int == 'O2-O2':
            oxygen_dimer_vertical_profile = 1e-46*atm_arr[:, atm_col_id_dict['O2']]**2
            gas_int_dict.update({gas_int:interp1d(altitude, oxygen_dimer_vertical_profile)})
        else:
            gas_int_dict.update({gas_int:interp1d(altitude, atm_arr[:, atm_col_id_dict[gas_int]])})
    #%%
    # Layer middle point might introduce error, check if possible to compute at layer boundary
    # Main loop
    profile = []
    
    start_time = time.time()
    for l in range(len(altitude) - 1):
        print('\nComputing {} {}/{}'.format(['layer', 'level'][slab_midpoint_flag], l + 1, len(altitude) - 1))
        
        # Layer altitude and size
        lz = np.abs(altitude[l] - altitude[l+1])
        
        if slab_midpoint_flag:
            # Compute properties to the middle point of layer
            lz_middle = np.min([altitude[l],altitude[l+1]]) + lz / 2
        else:
            lz_middle = np.min([altitude[l],altitude[l+1]])
        temp_middle = temperature_int(lz_middle)
        pressure_middle = pressure_int(lz_middle) / 1013.25
        
        # Compute number concentrations
        gas_concentration_dict = {}
        for gas_dil_val in gas_int_dict:
            number_concentration = gas_int_dict[gas_dil_val](lz_middle)
            gas_concentration_dict.update({gas_dil_val:number_concentration})
          
        # Compute mixing ratios
        total_gases = 0
        diluent_dict = {}
        for gas_percent in fetched_gases:
            total_gases += gas_concentration_dict[gas_percent]
            diluent_gas = gas_concentration_dict[gas_percent] / gas_concentration_dict['air']
            diluent_dict.update({gas_percent:diluent_gas})    
     
        for gas_percent in lookup_table_gases:
            total_gases += gas_concentration_dict[gas_percent]
            diluent_gas = gas_concentration_dict[gas_percent] / gas_concentration_dict['air']
            diluent_dict.update({gas_percent:diluent_gas}) 
        
        diluent_air = (gas_concentration_dict['air'] - total_gases) / gas_concentration_dict['air']
        diluent_dict.update({'air':diluent_air})
        
        # Unit [cm-1]
        abs_coef = 0
        
        for compute_gas in fetched_gases:
            print(f"Computing gas: {compute_gas}")
            if use_xsec_db:
                nu_high, coef_high = database_interpolation(compute_gas,nu_min,nu_max,nu_resolution,
                                   np.array([temp_middle]),np.array([pressure_middle]),np.array([diluent_dict[compute_gas]]))
                if not type(coef_high) == type(None):
                    coef_high = coef_high.ravel()
            else:
                coef_high = None
            if type(coef_high) == type(None):
                # if compute_gas not in fetched_gases:
                    # fetch_lines(compute_gas, nu_min, nu_max)
                    # fetched_gases.append(compute_gas)
                nu_high, coef_high = broadening_func(
                                               SourceTables=[compute_gas],
                                               Diluent={'air': 1 - diluent_dict[compute_gas],
                                                        #'air': diluent_dict['air'],
                                                        'self': diluent_dict[compute_gas]},
                                               WavenumberStep=nu_resolution,
                                               WavenumberRange=[nu_min, nu_max],
                                               Environment={'T':temp_middle,
                                                            'p':pressure_middle},
                                               HITRAN_units=True)
                if use_xsec_db:
                    input_xsec_data(compute_gas,np.array([temp_middle]),
                        np.array([pressure_middle]),np.array([diluent_dict[compute_gas]]),
                        nu_min,nu_max,nu_resolution,nu_high,coef_high)
            abs_coef += coef_high * gas_concentration_dict[compute_gas]
        for compute_gas in lookup_table_gases:
            if compute_gas == 'O3':
                print('Computing table O3!')
                abs_coef += ozone_cross_section(1 / (nu_high * 1e-7), temp_middle, table_O3) * gas_concentration_dict[compute_gas]
            elif compute_gas == 'NO2':
                print('Computing table NO2!')
                abs_coef += nitrogen_oxide_cross_section(1 / (nu_high * 1e-7), temp_middle, table_NO2) * gas_concentration_dict[compute_gas]
            elif compute_gas == 'O2-O2':
                print('Computing table O2-O2!')
                abs_coef += oxygen_dimer(1 / (nu_high * 1e-7), table_O2_O2) * gas_concentration_dict[compute_gas]
            else:
                print('Oops! Table gas not implemented yet!')
        optical_path = abs_coef * lz * 1e5
        profile.append(optical_path)
    
    if check_wavelength_limits:
        try:
            kurucs_full_wl = np.loadtxt(f'{libradtranpath}/data/solar_flux/kurudz_full.dat', comments='#', usecols=0)
            kurucs_lims = np.where(np.logical_and(kurucs_full_wl >= wl_range_nm[0], kurucs_full_wl <= wl_range_nm[-1]))[0]
            
            kurucs_min = kurucs_full_wl[kurucs_lims[0]]
            kurucs_max = kurucs_full_wl[kurucs_lims[-1]]
            
            print('When using file kurudz_full.dat in libRadtran')
            print(f'Minimum wavelength is: {kurucs_min}')
            print(f'Maximum wavelength is: {kurucs_max}')
    
        except (OSError, FileNotFoundError):
            print('File kurudz_full.dat not found in "<libRadtran_path>/data/solar_flux/"')
            print('Cannot provide limits for non-reflectivity calculations!')
                
    if create_mol_abs_netcdf_file:
        
        # This netcdf file is for libRadtran
        if atmosphere_id == "custom":
            atmos_str = 'custom'
        else:
            atmos_str = atmosphere_dict[atmosphere_id]
        molabs_fname = 'uvspec_{}_{}.nc'.format(atmos_str, name_suffix)
        with Dataset(molabs_fname, 'w', format='NETCDF4') as nc:
        
            # Global attributes
            nc.description = 'Molecular absorption optical depth file for libRadtran. Computed with HAPI2LIBIS.'
            nc.wavelength = 'Wavelength range {}-{} nm.'.format(wl_range_nm[0], wl_range_nm[-1])
            nc.use = 'mol_tau_file abs thisfile.nc'
        
            # Dimensions
            nc.createDimension('nlev', len(altitude))
            nc.createDimension('nlyr', len(altitude) - 1)
            nc.createDimension('nwvl', len(coef_high))
        
            # Variables
            nc_wvlmin = nc.createVariable('wvlmin','f8',())
            nc_wvlmax = nc.createVariable('wvlmax','f8',())
            nc_z      = nc.createVariable('z','f8',('nlev'))
            nc_wvl    = nc.createVariable('wvl','f8',('nwvl'))
            nc_tau    = nc.createVariable('tau','f8',('nlyr', 'nwvl'))
            
            # Attributes
            nc_wvlmin.longname = 'Minimum wavelength'
            nc_wvlmax.longname = 'Maximum wavelength'
            nc_z.longname      = 'Altitude'
            nc_wvl.longname    = 'Wavelength'
            nc_tau.longname    = 'Absorption optical depth of layer'    
            
            # Units
            nc_wvlmin.units = 'nm'
            nc_wvlmax.units = 'nm'
            nc_z.units      = 'km'
            nc_wvl.units    = 'nm'
            nc_tau.units    = 'unitless'
            
            
            # Reshape for libRadtran
            wl_num2nm = np.flip(1 / (nu_high * 1e-7))
            # transmittance_nm = np.flip(hapi_dict['transmittance'])
            opticalPath_nm = np.fliplr(profile)
            
            if boundary_flag:
                wl_num2nm[0], wl_num2nm[-1] = wl_range_nm[0], wl_range_nm[-1]
            
            # Assign values
            nc_wvlmin[:] = wl_range_nm[0]
            nc_wvlmax[:] = wl_range_nm[-1]
            nc_z[:]      = altitude
            nc_wvl[:]    = wl_num2nm
            nc_tau[:, :] = opticalPath_nm
        print('Saved mol.abs. file "%s"' % molabs_fname)
    print('HAPI2LIBIS done!')

