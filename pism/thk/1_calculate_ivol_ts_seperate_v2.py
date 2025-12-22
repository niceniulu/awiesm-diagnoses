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


########## calculate seperately and write to a file


# exps = [
#         'tran20-32k-sed30',
#         'tran20-32k-new04',
#         'tran20-32k-noEIS',
#         'tran20-32k-sia5',
#        ]
#
# datas = [ '../3_thk_ts/data/'+i+'_thk.nc'  for i in exps[:] ]

exps = ['trans']
datas = ['thk_topg_32k-0.nc',]

datas

fmask = '/Users/lniu/working/pool/mask_pism.nc'
farea = '/Users/lniu/working/pool/gridarea_pism.nc'

# ### define
ocean_area = 3.625e14  #m^2
# resolution = 20000  #m
rhoice = 910 #kg/m^3
rhosea = 1028 #kg/m^3
rr = rhoice/rhosea /ocean_area


########################################


with nc.Dataset(fmask) as ff:
    mask = ff.variables['mask'][:]

with nc.Dataset(farea) as ff:
    gridarea = ff.variables['cell_area'][:]


ones = np.zeros(mask.shape) + 1


for i in range(0,len(exps)):
    ### output files
    dataset = nc.Dataset('pism_'+exps[i]+'_sep_ts.nc','w',format='NETCDF4_CLASSIC')
    dataset.createDimension('years', None)
    yy = dataset.createVariable('years', np.float64, ('years',))
    # yy[:] = years[:]



    ######## input files
    fname = datas[i]
    with nc.Dataset(fname) as ff:
        thk = ff.variables['thk'][:]
        area = np.where(thk>0.01, 1., 0.)
        time = ff.variables['time'][:]
        yy[:] = time/365./3600./24.

    ##### Greenland
    var1 = dataset.createVariable('ivol_grn', np.float64, ('years',))
    var2 = dataset.createVariable('iarea_grn', np.float64, ('years',))
    # volume
    masknew = np.ma.masked_where(mask != 0, ones)
    aaa = masknew * thk * gridarea * rr
    sle_grn = np.sum(np.sum(aaa,axis=1), axis=1)
    # area
    bbb = masknew * area * gridarea
    iarea_grn = np.sum(np.sum(bbb,axis=1), axis=1)

    var1[:] = sle_grn[:]
    var2[:] = iarea_grn[:]


    ### North America
    var1 = dataset.createVariable('ivol_na', np.float64, ('years',))
    var2 = dataset.createVariable('iarea_na', np.float64, ('years',))

    condition = (mask != 1 ) & ( mask !=  2)  & ( mask !=  3)  & ( mask !=  4) & ( mask !=  5)
    masknew = np.ma.masked_where( condition , ones)
    aaa = masknew * thk * gridarea * rr
    sle_na = np.sum(np.sum(aaa,axis=1), axis=1)
    # area
    bbb = masknew * area * gridarea
    iarea_na = np.sum(np.sum(bbb,axis=1), axis=1)

    var1[:] = sle_na[:]
    var2[:] = iarea_na[:]

    # print("The index:", np.where((sle_na>83) & (sle_na<86.5)))


    ## Euasia
    var1 = dataset.createVariable('ivol_eu', np.float64, ('years',))
    var2 = dataset.createVariable('iarea_eu', np.float64, ('years',))

    condition = (mask != 16 ) & (mask != 17 ) &(mask != 18 ) &(mask != 28 ) &(mask != 29 ) & (mask != 46 )  &(mask != 30 )
    masknew = np.ma.masked_where( condition , ones)
    aaa = masknew * thk * gridarea * rr
    sle_eu = np.sum(np.sum(aaa,axis=1), axis=1)
    # area
    bbb = masknew * area * gridarea
    iarea_eu = np.sum(np.sum(bbb,axis=1), axis=1)

    var1[:] = sle_eu[:]
    var2[:] = iarea_eu[:]

    dataset.close()
