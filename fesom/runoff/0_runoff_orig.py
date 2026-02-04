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





###################### define
meshpath = '/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'

# kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}01.01.nc"}
kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}.nc"}



datapath='../../../outdata/fesom/'
ystart = 3000
yend = 4200
foutname = 'runoff_orig.nc'



#####################################################################################################
#################
# read data
mesh = pf.load_mesh(meshpath)
elem2=mesh.elem[mesh.no_cyclic_elem,:]

runoff = pf.get_data(datapath, 'runoff', range(ystart,yend), mesh, how="ori", compute=False, **kwargs )  # m/s

###### node areas
with nc.Dataset(meshpath+'fesom.mesh.diag.nc', 'r') as ff:
    nod_area = ff.variables['nod_area'][0,:]

### annual mean
datarunoff = runoff.groupby("time.year").mean(dim="time").values

nyear = datarunoff.shape[0]



#########################
# write to file
#########################


fout = nc.Dataset(foutname, 'w',format='NETCDF4_CLASSIC')
fout.createDimension('time', None)
fout.createDimension('nod2', datarunoff.shape[1])

timeout = fout.createVariable('time',np.float32,('time',))
timeout.units = 'years since 0001-01-01'
years = np.arange(ystart,yend)
timeout[:] = years[:]


datanew = fout.createVariable('runoff', np.float32, ('time','nod2'))
datanew.units = ''
datanew[:] = datarunoff[:]



fout.close()





#-
