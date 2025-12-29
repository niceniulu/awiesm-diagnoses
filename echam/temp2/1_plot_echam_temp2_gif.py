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

fname = 'tran-17k-hosing_echam_temp2_summer.nc'


dyear = 10

os.system("rm -r plots")
os.system("mkdir -p  plots")


### read data #############
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")

ds = xr.open_dataset(fname)
data = ds['temp2'].values - 273.15
years = ds['time'].values
lon = ds['lon'].values
lat = ds['lat'].values

nyear = data.shape[0]

datamean = np.mean(data[0:5,:],axis=0)

##########
# cyclic grids
data, lon = add_cyclic_point(data, coord=lon)
datamean = add_cyclic_point(datamean)

lons, lats = np.meshgrid(lon,lat)



#____ plotting _________
minval = -35
maxval = 35
nbreaks = 71
breaks = np.linspace(minval ,maxval ,nbreaks)
color = plt.cm.Spectral_r  #cmocean.cm.thermal   #haline

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

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(1, 1, 1,projection=proj)
    cs = ax.contourf(lons, lats, np.mean(data[i:i+5,:],axis=0) , extend='both', cmap=color,  levels=breaks)
    ax.set_extent([-180,180, -90,90], crs=proj)

    # iceyear = i*20 - 32000
    theyear = years[i]
    ax.set_title('Surface temperature year = '+str(theyear), fontsize=15)
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


    cax = plt.axes([0.1,0.05,0.85,0.03])
    cbar = plt.colorbar(cs,cax=cax, orientation='horizontal', fraction=0.02, extend='both')
    cbar.set_label(label='temp2', size=13, ) #weight='bold')
    cbar.ax.tick_params(labelsize=15)  #labelsize=15
    # cbar.set_ticks(np.arange(minval,maxval+0.1,0.25))

    fig.tight_layout()

    filename = f'plots/plots_{i+1:03d}.png'
    # fig.text(0.45,0.95, 'year='+str(i), fontsize=20)
    plt.savefig(filename,bbox_inches='tight',)
    plt.close()  # Close the figure to save memory


    image = imageio.imread(filename)
    frames.append(image)



gifname = "animated_temp2.gif"
# Save to GIF
imageio.mimsave(gifname, frames, duration=0.3)  # duration in seconds per frame

from IPython.display import Image

# Display the GIF inline
Image(filename=gifname)




#____ plotting _________
minval = -25
maxval = 25
nbreaks = 51
breaks = np.linspace(minval ,maxval ,nbreaks)
color = plt.cm.bwr   #cmocean.cm.phase  #haline

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

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(1, 1, 1,projection=proj)
    cs = ax.contourf(lons, lats, np.mean(data[i:i+5,:],axis=0)-datamean , extend='both', cmap=color, levels=breaks)
    ax.set_extent([-180,180, -90,90], crs=proj)

    # iceyear = i*20 - 32000
    theyear = years[i]
    ax.set_title('SAT anomaly  year = '+str(theyear), fontsize=15)
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


    cax = plt.axes([0.1,0.05,0.85,0.03])
    cbar = plt.colorbar(cs,cax=cax, orientation='horizontal', fraction=0.02, extend='both')
    cbar.set_label(label='temp2 anomaly', size=13, ) #weight='bold')
    cbar.ax.tick_params(labelsize=15)  #labelsize=15
    # cbar.set_ticks(np.arange(minval,maxval+0.1,0.25))

    fig.tight_layout()

    filename = f'plots/plots_{i+1:03d}_anomaly.png'   ############## define !!!
    # fig.text(0.45,0.95, 'year='+str(i), fontsize=20)
    plt.savefig(filename,bbox_inches='tight',)
    plt.close()  # Close the figure to save memory


    image = imageio.imread(filename)
    frames.append(image)



gifname = "animated_temp2_anomaly.gif"                    ############## define !!!
# Save to GIF
imageio.mimsave(gifname, frames, duration=0.3)  # duration in seconds per frame

from IPython.display import Image

# Display the GIF inline
Image(filename=gifname)
