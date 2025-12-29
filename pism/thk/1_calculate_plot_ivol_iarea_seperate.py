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


# ### define
ocean_area = 3.625e14  #m^2
# resolution = 20000  #m
rhoice = 910 #kg/m^3
rhosea = 1028 #kg/m^3
rr = rhoice/rhosea /ocean_area



## read ice sheet mask file
fmask = '/home/a/a270075/ba0989/pool/pism/grids/mask_pism_ice_sheet_seperate.nc'
with nc.Dataset(fmask,'r') as ff:
    mask = ff.variables['mask'][:]

farea = '/home/a/a270075/ba0989/pool/pism/grids/gridarea_pism.nc'
with nc.Dataset(farea) as ff:
    gridarea = ff.variables['cell_area'][:]

fname = 'tran-17k-hosing_thk_topg.nc'
with nc.Dataset(fname, 'r') as ff:
    x = ff.variables['x'][:]
    y = ff.variables['y'][:]
    thk = ff.variables['thk'][:,:,:]
    time = ff.variables['time'][:]


years =  time/365./3600./24.  #np.arange(-24000, 0, 100)


#  ## GIS
# data = np.where( (mask==100), thk, 0)
# sle_GIS = calculate_thk_to_sle(data)
# ## LIS
# data = np.where( (mask==200), thk, 0)
# sle_LIS = calculate_thk_to_sle(data)
# ## CIS
# data = np.where( (mask==300), thk, 0)
# sle_CIS = calculate_thk_to_sle(data)
# ## EIS
# data = np.where( (mask==400), thk, 0)
# sle_EIS = calculate_thk_to_sle(data)

### output files
dataset = nc.Dataset('pism_ivol_iarea_sep_ts.nc','w',format='NETCDF4_CLASSIC')
dataset.createDimension('years', None)
yy = dataset.createVariable('years', np.float64, ('years',))
yy[:] = years[:]


## GIS
var1 = dataset.createVariable('ivol_GIS', np.float64, ('years',))
var2 = dataset.createVariable('iarea_GIS', np.float64, ('years',))

data = np.where( (mask==100), thk, 0)
data2 = np.where( (mask==100)&(thk>0.1), thk-thk+1., 0)
#volume
aaa = data * gridarea * rr
sle_GIS = np.sum(np.sum(aaa,axis=1), axis=1)
# area
bbb = data2 * gridarea
iarea_GIS = np.sum(np.sum(bbb,axis=1), axis=1)

var1[:] = sle_GIS[:]
var2[:] = iarea_GIS[:]


## LIS
var1 = dataset.createVariable('ivol_LIS', np.float64, ('years',))
var2 = dataset.createVariable('iarea_LIS', np.float64, ('years',))

data = np.where( (mask==200), thk, 0)
data2 = np.where( (mask==200)&(thk>0.1), thk-thk+1., 0)
#volume
aaa = data * gridarea * rr
sle_LIS = np.sum(np.sum(aaa,axis=1), axis=1)
# area
bbb = data2 * gridarea
iarea_LIS = np.sum(np.sum(bbb,axis=1), axis=1)

var1[:] = sle_LIS[:]
var2[:] = iarea_LIS[:]



## CIS
var1 = dataset.createVariable('ivol_CIS', np.float64, ('years',))
var2 = dataset.createVariable('iarea_CIS', np.float64, ('years',))

data = np.where( (mask==300), thk, 0)
data2 = np.where( (mask==300)&(thk>0.1), thk-thk+1., 0)
#volume
aaa = data * gridarea * rr
sle_CIS = np.sum(np.sum(aaa,axis=1), axis=1)
# area
bbb = data2 * gridarea
iarea_CIS = np.sum(np.sum(bbb,axis=1), axis=1)

var1[:] = sle_CIS[:]
var2[:] = iarea_CIS[:]



## EIS
var1 = dataset.createVariable('ivol_EIS', np.float64, ('years',))
var2 = dataset.createVariable('iarea_EIS', np.float64, ('years',))

data = np.where( (mask==400), thk, 0)
data2 = np.where( (mask==400)&(thk>0.1), thk-thk+1., 0)
#volume
aaa = data * gridarea * rr
sle_EIS = np.sum(np.sum(aaa,axis=1), axis=1)
# area
bbb = data2 * gridarea
iarea_EIS = np.sum(np.sum(bbb,axis=1), axis=1)

var1[:] = sle_EIS[:]
var2[:] = iarea_EIS[:]


dataset.close()

total = sle_LIS + sle_CIS + sle_EIS + sle_GIS



##########################################################################
##########################################################################
## plotting ####
fig, axs = plt.subplots(6,1, sharex=True, figsize=(12,16) )
axs[0].plot(years , sle_LIS, label='LIS',color='red')
axs[0].legend()
axs[1].plot(years , sle_CIS, label='CIS',color='brown')
axs[1].legend()
axs[2].plot(years , sle_EIS, label='EIS',color='blue')
axs[2].legend()
axs[3].plot(years , sle_GIS, label='GIS',color='green')
axs[3].legend()
axs[4].plot(years , total, label='Total',color='black')
axs[4].legend()





# axs[0].set_xlim(-24000, 0)
# axs[0].set_ylim(-10, 120)

# axs[0].set_xlabel("Model year", fontsize=10)
axs[0].set_ylabel("ivol (m SLE)", fontsize=10)

axs[5].plot(years , dsle_per_year(sle_LIS, years), label='LIS',color='red')
axs[5].plot(years , dsle_per_year(sle_CIS, years), label='CIS',color='brown')
axs[5].plot(years , dsle_per_year(sle_EIS, years), label='EIS',color='blue')
axs[5].plot(years , dsle_per_year(sle_GIS, years), label='GIS',color='green')
axs[5].plot(years , dsle_per_year(total, years), label='Total',color='black')

axs[5].set_xlabel("Model year", fontsize=10)
axs[5].set_ylabel("dSLE (m/year)", fontsize=10)

# axs[1].set_ylim(-0.03, 0.01)

plt.savefig('1_plot_ivol_seperate.png',bbox_inches='tight',)
