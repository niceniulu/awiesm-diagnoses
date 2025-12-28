#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
import pyfesom2 as pf
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation,TriAnalyzer
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cmocean

from pyproj import Proj, transform



meshpath = '/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'
fname = 'a_ice_oneyear.nc'




mesh = pf.load_mesh(meshpath)
elem2=mesh.elem[mesh.no_cyclic_elem,:]



### read data #############
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
ds = xr.open_dataset(fname)
data = ds['a_ice'].values


#____ plot sea ice _________
proj = ccrs.PlateCarree()
crs_target = ccrs.NorthPolarStereo(central_longitude= -45,true_scale_latitude=70.,globe=None)
crs_target2 = ccrs.SouthPolarStereo(central_longitude= -45,true_scale_latitude=-70.,globe=None)
print(crs_target.proj4_params)
print(crs_target2.proj4_params)

breaks = np.linspace(1e-8,1,41)
months=[3,9]

lonticks = np.arange(-180.,180.1,60.)
latticks = np.arange(-90.,90.1,30.)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()


############### polar projection  ###############

fig = plt.figure(figsize=(12,8))


for i in range(0, len(months)):
    ax = fig.add_subplot(2, len(months), i+1,projection=crs_target)
    cs = plt.tricontourf(mesh.x2, mesh.y2, elem2, data[months[i]-1,:],extend='neither',cmap=cmocean.cm.ice, levels=breaks, transform=proj)
    ax.set_extent([-180,180, 30,90], crs=proj)
    ax.coastlines()
    ax.set_title('SIC(%) mon='+str(months[i]))
    plt.colorbar(cs)

    ax2 = fig.add_subplot(2, len(months), i+len(months)+1,projection=crs_target2)
    cs2 = plt.tricontourf(mesh.x2, mesh.y2, elem2, data[months[i]-1,:],extend='neither',cmap=cmocean.cm.ice, levels=breaks, transform=proj)
    ax2.set_extent([-180,180, -30,-90], crs=proj)
    ax2.coastlines()
    ax2.set_title('SIC(%) mon='+str(months[i]))
    plt.colorbar(cs2)


fig.tight_layout()


plt.savefig('2_seaice_polar.png')
plt.show(block=False)
plt.pause(3)
plt.close()
