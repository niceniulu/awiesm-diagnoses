#!/usr/bin/env python3
# coding: utf-8


import numpy as np
import netCDF4 as nc
import pyfesom2 as pf
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

import cmocean
import os


# def calculate_thk_to_sle(thk, resolution = 20000 ):
#     ocean_area = 3.625e14  #m^2
#     # resolution = 20000  #m
#     rhoice = 910 #kg/m^3
#     rhosea = 1028 #kg/m^3
#     rr = rhoice/rhosea /ocean_area
#     # units: m
#     aaa = thk * resolution * resolution * rr
#     sle = np.sum(np.sum(aaa,axis=1), axis=1)
#     return sle


def dsle_per_year(data, years):
    from scipy.interpolate import UnivariateSpline

    # Fit a spline to the data
    spl = UnivariateSpline(years, data, s=0)
    # Compute the derivative of the spline
    derivative = spl.derivative()(years)

    return derivative


def kgm2year_to_sv(data, resolution=20000):
    rho = 1000   # kg/m^3
    out = data*resolution*resolution/rho/(365.*24*3600)/1e6
    return out


## read ice sheet mask file
fmask = '/home/a/a270075/ba0989/pool/pism/grids/mask_pism_ice_sheet_seperate.nc'
with nc.Dataset(fmask,'r') as ff:
    mask = ff.variables['mask'][:]



fname = 'tran-17k-hosing_extra.nc'
with nc.Dataset(fname, 'r') as ff:
    x = ff.variables['x'][:]
    y = ff.variables['y'][:]
    # thk = ff.variables['thk'][:]
    basal = ff.variables['tendency_of_ice_amount_due_to_basal_mass_flux'][:]
    discharge = ff.variables['tendency_of_ice_amount_due_to_discharge'][:]
    surface = ff.variables['tendency_of_ice_amount_due_to_surface_mass_flux'][:]
    calving = ff.variables['tendency_of_ice_amount_due_to_calving'][:]
    time = ff.variables['time'][:]

years =  time/365./3600./24.

################################################################################
### output files
################################################################################
dataset = nc.Dataset('pism_massbalance_sep_ts.nc','w',format='NETCDF4_CLASSIC')
dataset.createDimension('years', None)
yy = dataset.createVariable('years', np.float64, ('years',))
yy[:] = years[:]



## GIS
var1 = dataset.createVariable('positive_GIS', np.float64, ('years',))
var2 = dataset.createVariable('negative_GIS', np.float64, ('years',))

surface_GIS = np.where( (mask==100), surface, 0)
data = np.where(surface_GIS>0., surface_GIS, 0. )
positive_GIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))
data = np.where(surface_GIS<0., surface_GIS, 0. )
negative_GIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))


var1[:] = positive_GIS[:]
var2[:] = negative_GIS[:]



## LIS
var1 = dataset.createVariable('positive_LIS', np.float64, ('years',))
var2 = dataset.createVariable('negative_LIS', np.float64, ('years',))

surface_LIS = np.where( (mask==200), surface, 0)
data = np.where(surface_LIS>0., surface_LIS, 0. )
positive_LIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))
data = np.where(surface_LIS<0., surface_LIS, 0. )
negative_LIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))


var1[:] = positive_LIS[:]
var2[:] = negative_LIS[:]


## EIS
var1 = dataset.createVariable('positive_EIS', np.float64, ('years',))
var2 = dataset.createVariable('negative_EIS', np.float64, ('years',))

surface_EIS = np.where( (mask==400), surface, 0)
data = np.where(surface_EIS>0., surface_EIS, 0. )
positive_EIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))
data = np.where(surface_EIS<0., surface_EIS, 0. )
negative_EIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))


var1[:] = positive_EIS[:]
var2[:] = negative_EIS[:]


## CIS
var1 = dataset.createVariable('positive_CIS', np.float64, ('years',))
var2 = dataset.createVariable('negative_CIS', np.float64, ('years',))

surface_CIS = np.where( (mask==300), surface, 0)
data = np.where(surface_CIS>0., surface_CIS, 0. )
positive_CIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))
data = np.where(surface_CIS<0., surface_CIS, 0. )
negative_CIS = kgm2year_to_sv(np.sum(np.sum(data, axis=1), axis=1))


var1[:] = positive_CIS[:]
var2[:] = negative_CIS[:]


dataset.close()



## plotting ####
fig, axs = plt.subplots(5,1, sharex=True, figsize=(12,15))

axs[0].plot(years, positive_LIS, label='positive smb',color='red')
axs[0].plot(years, negative_LIS, label='negative smb',color='blue')
axs[0].legend()

# axs[0].set_xlim(-24000, 0)
# axs[0].set_ylim(-0.2, 0.2)
axs[0].set_ylabel("LIS fwf (Sv)", fontsize=10)

axs[1].plot(years, positive_EIS, label='positive smb',color='red')
axs[1].plot(years, negative_EIS, label='negative smb',color='blue')
axs[1].legend()

# axs[1].set_xlim(-24000, 0)
# axs[1].set_ylim(-0.2, 0.2)
axs[1].set_ylabel("EIS fwf (Sv)", fontsize=10)


axs[2].plot(years, positive_CIS, label='positive smb',color='red')
axs[2].plot(years, negative_CIS, label='negative smb',color='blue')
axs[2].legend()

# axs[2].set_xlim(-24000, 0)
# axs[2].set_ylim(-0.2, 0.2)
axs[2].set_ylabel("CIS fwf (Sv)", fontsize=10)


axs[3].plot(years, positive_GIS, label='positive smb',color='red')
axs[3].plot(years, negative_GIS, label='negative smb',color='blue')
axs[3].legend()

# axs[3].set_xlim(-24000, 0)
# axs[3].set_ylim(-0.2, 0.2)
axs[3].set_ylabel("GIS fwf (Sv)", fontsize=10)



axs[4].plot(years, positive_LIS + positive_EIS + positive_CIS + positive_GIS, label='positive smb',color='red')
axs[4].plot(years, negative_LIS + negative_EIS + negative_CIS+ negative_GIS , label='negative smb',color='blue')
axs[4].legend()

# axs[4].set_xlim(-24000, 0)
# axs[4].set_ylim(-0.2, 0.2)
axs[4].set_ylabel("Total fwf (Sv)", fontsize=10)

plt.savefig('2_smb_budget.png',bbox_inches='tight',)
