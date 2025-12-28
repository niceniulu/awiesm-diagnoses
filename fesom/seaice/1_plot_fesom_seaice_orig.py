#!/usr/bin/env python3

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

from  scipy  import   stats



meshpath = '/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'
fname = 'a_ice.fesom.orig.nc'

dyear = 10

os.system("rm -r plots")
os.system("mkdir -p  plots")


mesh = pf.load_mesh(meshpath)
elem2=mesh.elem[mesh.no_cyclic_elem,:]



### read data #############
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
ds = xr.open_dataset(fname)
data = ds['a_ice'].values
data2 = data.reshape(-1, 12, data.shape[1])   # seperate to months

years = ds['time'].values
print(data.shape, mesh.x2.shape, elem2.shape)
print(data2.shape)


nyear = data2.shape[0]






######################################################################################
######################################################################################
#____ plotting _________ absolute values
minval = 1e-8
maxval = 1
nbreaks = 41
breaks = np.linspace(minval ,maxval ,nbreaks)
color = cmocean.cm.ice

################
proj = ccrs.PlateCarree()
crs_target = ccrs.NorthPolarStereo(central_longitude= -45,true_scale_latitude=70.,globe=None)
crs_target2 = ccrs.SouthPolarStereo(central_longitude= -45,true_scale_latitude=-70.,globe=None)
print(crs_target.proj4_params)
print(crs_target2.proj4_params)

lonticks = np.arange(-180.,180.1,60.)
latticks = np.arange(-90.,90.1,30.)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()


# ############### normal projection  ###############

## for gif files
import imageio
frames = []


for i in range(0, nyear, dyear):
    print(i)

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(2, 1, 1,projection=proj)
    cs = ax.tricontourf(mesh.x2, mesh.y2, elem2, np.mean(data2[i:i+5,2, :],axis=0) ,extend='neither', cmap=color, levels=breaks)
    ax.set_extent([-180,180, -90,90], crs=proj)

    # iceyear = i*20 - 32000
    theyear = i
    ax.set_title('a_ice mon=3,  year = '+str(theyear), fontsize=15)
    ax.set_xlabel("Longitude (°)", fontsize=15)
    ax.set_ylabel("Latitude (°)", fontsize=15)

    ax.set_global()
    ax.coastlines()
    ax.set_yticks(latticks, crs=proj)
    ax.set_xticks(lonticks, crs=proj)
    ax.tick_params(axis='x', which='major', direction='out',size=5, width=1.5, labelsize=14)
    ax.tick_params(axis='y', which='major', direction='out',size=5, width=1.5, labelsize=14)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)



    #########################################
    ax = fig.add_subplot(2, 1, 2,projection=proj)
    cs = ax.tricontourf(mesh.x2, mesh.y2, elem2, np.mean(data2[i:i+5,8, :],axis=0) ,extend='neither', cmap=color, levels=breaks)
    ax.set_extent([-180,180, -90,90], crs=proj)

    # iceyear = i*20 - 32000
    theyear = i
    ax.set_title('a_ice mon=9,  year = '+str(theyear), fontsize=15)
    ax.set_xlabel("Longitude (°)", fontsize=15)
    ax.set_ylabel("Latitude (°)", fontsize=15)

    ax.set_global()
    ax.coastlines()
    ax.set_yticks(latticks, crs=proj)
    ax.set_xticks(lonticks, crs=proj)
    ax.tick_params(axis='x', which='major', direction='out',size=5, width=1.5, labelsize=14)
    ax.tick_params(axis='y', which='major', direction='out',size=5, width=1.5, labelsize=14)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)


    cax = plt.axes([0.2,0.05,0.6,0.03])
    cbar = plt.colorbar(cs,cax=cax, orientation='horizontal', fraction=0.02, extend='both')
    cbar.set_label(label='a_ice', size=13, ) #weight='bold')
    cbar.ax.tick_params(labelsize=15)  #labelsize=15
    # cbar.set_ticks(np.arange(minval,maxval+0.1,0.25))

    fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.13)
    # fig.tight_layout()



    ####################################### plot to png and combine to gif
    #######################################
    filename = f'plots/plots_{i+1:04d}.png'
    # fig.text(0.45,0.95, 'year='+str(i), fontsize=20)
    plt.savefig(filename,bbox_inches='tight',)
    plt.close()  # Close the figure to save memory


    image = imageio.imread(filename)
    frames.append(image)



gifname = "animated_a_ice.gif"
# Save to GIF
imageio.mimsave(gifname, frames, duration=0.3)  # duration in seconds per frame

from IPython.display import Image

# Display the GIF inline
Image(filename=gifname)
