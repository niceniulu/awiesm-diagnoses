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


meshpath = '/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'
datapath = './'
fnames = ['runoff_orig.nc',
           ]



mesh = pf.load_mesh(meshpath)
elem2=mesh.elem[mesh.no_cyclic_elem,:]
###### node areas
with nc.Dataset(meshpath+'fesom.mesh.diag.nc', 'r') as ff:
    nod_area = ff.variables['nod_area'][0,:]

themask = np.zeros(nod_area.shape)
ones = np.zeros(nod_area.shape)

############ define mask
mask_arc = pf.get_mask(mesh, "Arctic_Basin")   # 1. Arctic
# pf.tplot(mesh, mask_arc*nod_area)
# plt.show(block=False)
# plt.pause(3)

### define new atlantic basin, over north of xx degree

condition = (mesh.y2 >= 50) &(mesh.y2 <= 80) & (mesh.x2 >= -90) & (mesh.x2 <= -15)   #(mesh.x2 <= 130)
mask = np.where(condition, True, False)
mask_atlna= mask & ~mask_arc

# mask_all = mask | mask_arc  # 3. Atlantic + Arctic
# pf.tplot(mesh, mask_atlna*nod_area)
# plt.show(block=False)
# plt.pause(3)
themask = np.where(mask_atlna, 1., themask)


condition = (mesh.y2 >= 50) &(mesh.y2 <= 80) & (mesh.x2 >= -15) & (mesh.x2 <= 130)
mask = np.where(condition, True, False)
mask_atleu= mask & ~mask_arc

# pf.tplot(mesh, mask_atleu*nod_area)
# plt.show(block=False)
# plt.pause(3)
themask = np.where(mask_atleu, 2., themask)


################## mask arctic and NA sector
condition =  (mesh.x2 >= -170) & (mesh.x2 <= -15)
mask = np.where(condition, True, False)
mask_arcna = mask & mask_arc  # 2. Atlantic only north of 55

# pf.tplot(mesh, mask_arcna*nod_area)
# plt.show()
# plt.show(block=False)
# plt.pause(3)
themask = np.where(mask_arcna, 3., themask)


condition = ( (mesh.x2 >= -180) & (mesh.x2 < -170)) | ((mesh.x2 > -15) & (mesh.x2 <=180))
mask = np.where(condition, True, False)
mask_arceu = mask & mask_arc  # 2. Atlantic only north of 55

# pf.tplot(mesh, mask_arceu*nod_area)
# plt.show()
# plt.show(block=False)
# plt.pause(3)
themask = np.where(mask_arceu, 4., themask)


pf.tplot(mesh, themask)
plt.show()
plt.show(block=False)
plt.pause(3)



fout = nc.Dataset('themask.nc', 'w')
fout.createDimension('nod2', themask.shape[0])
tmp = fout.createVariable('mask', np.float32, ('nod2'))
print(tmp)
print(themask.shape)
tmp[:] = themask
fout.close()

# exit()

#########################
# write to file
#########################


for i in range(0,len(fnames)):
    with nc.Dataset(datapath + fnames[i], 'r') as ff:
        years = ff.variables['time'][:]
        runoff = ff.variables['runoff'][:]

        runoffarc_na = np.sum(runoff * mask_arcna * nod_area, axis=1)/1e6  # in Sv
        runoffarc_eu = np.sum(runoff * mask_arceu * nod_area, axis=1)/1e6

        runoffatl_na = np.sum(runoff * mask_atlna * nod_area, axis=1)/1e6
        runoffatl_eu = np.sum(runoff * mask_atleu * nod_area, axis=1)/1e6



    foutname = fnames[i].replace('.nc', '_timeseries.nc')

    fout = nc.Dataset(foutname, 'w',format='NETCDF4_CLASSIC')
    fout.createDimension('time', None)
    timeout = fout.createVariable('time',np.float32,('time',))
    timeout.units = 'years since 0001-01-01'
    timeout[:] = years[:]

    varout = fout.createVariable('runoffarc_na',np.float32,('time',))
    varout.units = 'Sv'
    varout[:] = runoffarc_na[:]

    varout = fout.createVariable('runoffarc_eu',np.float32,('time',))
    varout.units = 'Sv'
    varout[:] = runoffarc_eu[:]

    varout = fout.createVariable('runoffatl_na',np.float32,('time',))
    varout.units = 'Sv'
    varout[:] = runoffatl_na[:]

    varout = fout.createVariable('runoffatl_eu',np.float32,('time',))
    varout.units = 'Sv'
    varout[:] = runoffatl_eu[:]




    fout.close()

    # print(runoffatl_na[:])


    ##############  plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1,1)

    ax.plot(years, runoffarc_na, color='red', ls='-', label='North America - Arctic')
    ax.plot(years, runoffarc_eu, color='blue', ls='-', label='Eurasia - Arctic')

    ax.plot(years, runoffatl_na, color='red', ls='--', label='North America - Atlantic')
    ax.plot(years, runoffatl_eu, color='blue', ls='--', label='Eurasia - Atlantic')

    # ax.set_xlim(1, 50)
    # ax.set_ylim(-2, 2)
    ax.set_ylabel('runoff ()', fontsize=15)
    ax.set_title('runoff Arctic', fontsize=15,  loc='left')
    ax.tick_params(labelsize=13)

    ax.legend()

    plt.savefig('runoff_basin_ts.png')
    # plt.show(block=False)
    # plt.pause(3)
    plt.show()
