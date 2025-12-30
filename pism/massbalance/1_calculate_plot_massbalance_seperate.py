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
var1 = dataset.createVariable('surface_GIS', np.float64, ('years',))
var2 = dataset.createVariable('basal_GIS', np.float64, ('years',))
var3 = dataset.createVariable('discharge_GIS', np.float64, ('years',))
var4 = dataset.createVariable('calving_GIS', np.float64, ('years',))

surface_GIS = np.where( (mask==100), surface, 0)
water_surface_GIS = kgm2year_to_sv(np.sum(np.sum(surface_GIS, axis=1), axis=1))

basal_GIS = np.where( (mask==100), basal, 0)
water_basal_GIS = kgm2year_to_sv(np.sum(np.sum(basal_GIS, axis=1), axis=1))


discharge_GIS = np.where( (mask==100), discharge, 0)
water_discharge_GIS = kgm2year_to_sv(np.sum(np.sum(discharge_GIS, axis=1), axis=1))

calving_GIS = np.where( (mask==100), calving, 0)
water_calving_GIS = kgm2year_to_sv(np.sum(np.sum(calving_GIS, axis=1), axis=1))

var1[:] = water_surface_GIS[:]
var2[:] = water_basal_GIS[:]
var3[:] = water_discharge_GIS[:]
var4[:] = water_calving_GIS[:]


## LIS
var1 = dataset.createVariable('surface_LIS', np.float64, ('years',))
var2 = dataset.createVariable('basal_LIS', np.float64, ('years',))
var3 = dataset.createVariable('discharge_LIS', np.float64, ('years',))
var4 = dataset.createVariable('calving_LIS', np.float64, ('years',))

surface_LIS = np.where( (mask==200), surface, 0)
water_surface_LIS = kgm2year_to_sv(np.sum(np.sum(surface_LIS, axis=1), axis=1))

basal_LIS = np.where( (mask==200), basal, 0)
water_basal_LIS = kgm2year_to_sv(np.sum(np.sum(basal_LIS, axis=1), axis=1))


discharge_LIS = np.where( (mask==200), discharge, 0)
water_discharge_LIS = kgm2year_to_sv(np.sum(np.sum(discharge_LIS, axis=1), axis=1))

calving_LIS = np.where( (mask==200), calving, 0)
water_calving_LIS = kgm2year_to_sv(np.sum(np.sum(calving_LIS, axis=1), axis=1))

var1[:] = water_surface_LIS[:]
var2[:] = water_basal_LIS[:]
var3[:] = water_discharge_LIS[:]
var4[:] = water_calving_LIS[:]


## EIS
var1 = dataset.createVariable('surface_EIS', np.float64, ('years',))
var2 = dataset.createVariable('basal_EIS', np.float64, ('years',))
var3 = dataset.createVariable('discharge_EIS', np.float64, ('years',))
var4 = dataset.createVariable('calving_EIS', np.float64, ('years',))

surface_EIS = np.where( (mask==400), surface, 0)
water_surface_EIS = kgm2year_to_sv(np.sum(np.sum(surface_EIS, axis=1), axis=1))

basal_EIS = np.where( (mask==400), basal, 0)
water_basal_EIS = kgm2year_to_sv(np.sum(np.sum(basal_EIS, axis=1), axis=1))


discharge_EIS = np.where( (mask==400), discharge, 0)
water_discharge_EIS = kgm2year_to_sv(np.sum(np.sum(discharge_EIS, axis=1), axis=1))

calving_EIS = np.where( (mask==400), calving, 0)
water_calving_EIS = kgm2year_to_sv(np.sum(np.sum(calving_EIS, axis=1), axis=1))

var1[:] = water_surface_EIS[:]
var2[:] = water_basal_EIS[:]
var3[:] = water_discharge_EIS[:]
var4[:] = water_calving_EIS[:]


## CIS
var1 = dataset.createVariable('surface_CIS', np.float64, ('years',))
var2 = dataset.createVariable('basal_CIS', np.float64, ('years',))
var3 = dataset.createVariable('discharge_CIS', np.float64, ('years',))
var4 = dataset.createVariable('calving_CIS', np.float64, ('years',))

surface_CIS = np.where( (mask==300), surface, 0)
water_surface_CIS = kgm2year_to_sv(np.sum(np.sum(surface_CIS, axis=1), axis=1))

basal_CIS = np.where( (mask==300), basal, 0)
water_basal_CIS = kgm2year_to_sv(np.sum(np.sum(basal_CIS, axis=1), axis=1))


discharge_CIS = np.where( (mask==300), discharge, 0)
water_discharge_CIS = kgm2year_to_sv(np.sum(np.sum(discharge_CIS, axis=1), axis=1))

calving_CIS = np.where( (mask==300), calving, 0)
water_calving_CIS = kgm2year_to_sv(np.sum(np.sum(calving_CIS, axis=1), axis=1))

var1[:] = water_surface_CIS[:]
var2[:] = water_basal_CIS[:]
var3[:] = water_discharge_CIS[:]
var4[:] = water_calving_CIS[:]


dataset.close()



## plotting ####
fig, axs = plt.subplots(5,1, sharex=True, figsize=(12,15))

axs[0].plot(years, water_surface_LIS, label='smb',color='red')
axs[0].plot(years, water_calving_LIS, label='calving',color='blue')
axs[0].plot(years, water_discharge_LIS, label='discharge',color='lightblue')
axs[0].plot(years, water_basal_LIS, label='basal',color='black')
axs[0].legend()

# axs[0].set_xlim(-24000, 0)
# axs[0].set_ylim(-0.2, 0.2)
axs[0].set_ylabel("LIS fwf (Sv)", fontsize=10)

axs[1].plot(years, water_surface_EIS, label='smb',color='red')
axs[1].plot(years, water_calving_EIS, label='calving',color='blue')
axs[1].plot(years, water_discharge_EIS, label='discharge',color='lightblue')
axs[1].plot(years, water_basal_EIS, label='basal',color='black')
axs[1].legend()

# axs[1].set_xlim(-24000, 0)
# axs[1].set_ylim(-0.2, 0.2)
axs[1].set_ylabel("EIS fwf (Sv)", fontsize=10)


axs[2].plot(years, water_surface_CIS, label='smb',color='red')
axs[2].plot(years, water_calving_CIS, label='calving',color='blue')
axs[2].plot(years, water_discharge_CIS, label='discharge',color='lightblue')
axs[2].plot(years, water_basal_CIS, label='basal',color='black')
axs[2].legend()

# axs[2].set_xlim(-24000, 0)
# axs[2].set_ylim(-0.2, 0.2)
axs[2].set_ylabel("CIS fwf (Sv)", fontsize=10)


axs[3].plot(years, water_surface_GIS, label='smb',color='red')
axs[3].plot(years, water_calving_GIS, label='calving',color='blue')
axs[3].plot(years, water_discharge_GIS, label='discharge',color='lightblue')
axs[3].plot(years, water_basal_GIS, label='basal',color='black')
axs[3].legend()

# axs[3].set_xlim(-24000, 0)
# axs[3].set_ylim(-0.2, 0.2)
axs[3].set_ylabel("GIS fwf (Sv)", fontsize=10)



axs[4].plot(years, water_surface_LIS + water_surface_EIS + water_surface_CIS + water_surface_GIS, label='smb',color='red')
axs[4].plot(years, water_calving_LIS + water_calving_EIS + water_calving_CIS+ water_calving_GIS , label='calving',color='blue')
axs[4].plot(years, water_discharge_LIS + water_discharge_EIS + water_discharge_CIS+ water_discharge_GIS , label='discharge',color='lightblue')
axs[4].plot(years, water_basal_LIS + water_basal_EIS + water_basal_CIS+ water_basal_GIS, label='basal',color='black')
axs[4].legend()

# axs[4].set_xlim(-24000, 0)
# axs[4].set_ylim(-0.2, 0.2)
axs[4].set_ylabel("Total fwf (Sv)", fontsize=10)

plt.savefig('1_massbalance_budget.png',bbox_inches='tight',)
