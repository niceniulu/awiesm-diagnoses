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


def density_flux(datasss, datasst, datafh, datafw):
    import gsw
    ###############################################
    ####### density flux
    # = densityh + densitys =  alpha*Q/Cp + rho(0, T)*beta*fwf*sss/(1-sss)   # upward positive
    ###############################################
    tmp, alpha, beta = gsw.density.rho_alpha_beta(datasss, datasst, 0)
    rho = gsw.density.rho(0, datasst,  0)
    cp = gsw.cp_t_exact(datasss, datasst, 0.)
    print("cp", np.max(cp), np.min(cp) )

    densityH = alpha * datafh/ cp  # 1/K * w/m2 (J/s*m2) * 1/(J/kg*K)
    densityS = beta * rho * datafw * datasss/(1 - datasss/1000. )   #1/psu * kg/m3 * m/s * psu/(1-psu/1000)
    density = densityH + densityS

    return densityH, densityS, density





###################### define
meshpath = '/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'

# kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}01.01.nc"}
kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}.nc"}



datapath='../../../outdata/fesom/'
ystart = 3000
yend = 4200
foutname = 'density_flux_orig.nc'



#####################################################################################################
#################
# read data
mesh = pf.load_mesh(meshpath)
elem2=mesh.elem[mesh.no_cyclic_elem,:]


fw = pf.get_data(datapath, 'fw', range(ystart,yend), mesh, how="ori", compute=False, **kwargs )  # m/s
fh = pf.get_data(datapath, 'fh', range(ystart,yend), mesh, how="ori", compute=False, **kwargs )  # W/m2?
sss = pf.get_data(datapath, 'sss', range(ystart,yend), mesh, how="ori", compute=False, **kwargs )  #psu
sst = pf.get_data(datapath, 'sst', range(ystart,yend), mesh, how="ori", compute=False, **kwargs )  #degC

###### node areas
with nc.Dataset(meshpath+'fesom.mesh.diag.nc', 'r') as ff:
    nod_area = ff.variables['nod_area'][0,:]

### annual mean
datafw = fw.groupby("time.year").mean(dim="time").values
datafh = fh.groupby("time.year").mean(dim="time").values
datasss = sss.groupby("time.year").mean(dim="time").values
datasst = sst.groupby("time.year").mean(dim="time").values

print(datafw.shape)
nyear = datafw.shape[0]

densityH, densityS, density = density_flux(datasss, datasst, datafh, datafw)

#########################
# write to file
#########################


fout = nc.Dataset(foutname, 'w',format='NETCDF4_CLASSIC')
fout.createDimension('time', None)
fout.createDimension('nod2', datafw.shape[1])

timeout = fout.createVariable('time',np.float32,('time',))
timeout.units = 'years since 0001-01-01'
years = np.arange(ystart,yend)
timeout[:] = years[:]


density_heat = fout.createVariable('density_heat', np.float32, ('time','nod2'))
density_heat.units = 'kg/m2s'
density_heat[:] = densityH[:]

density_salt = fout.createVariable('density_salt', np.float32, ('time','nod2'))
density_salt.units = 'kg/m2s'
density_salt[:] = densityS[:]

dsss = fout.createVariable('sss', np.float32, ('time','nod2'))
dsss.units = 'psu'
dsss[:] = datasss[:]

dsst = fout.createVariable('sst', np.float32, ('time','nod2'))
dsst.units = 'degC'
dsst[:] = datasst[:]



fout.close()





#-
