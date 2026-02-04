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

mesh = pf.load_mesh(meshpath)
elem2=mesh.elem[mesh.no_cyclic_elem,:]



fname = 'runoff_orig.nc'


#####################################################################################################
#################
# read data
with nc.Dataset(fname, 'r') as ff:
    runoff = ff.variables['runoff'][:]
    time = ff.variables['time'][:]


nyear = len(time)
dt = 10



###############################################
## check plot 2d
###############################################
os.system("rm -rf  plots")
os.system("mkdir   plots")

#____ plotting _________
minval = 0
maxval = 1e-6
nbreaks = 51
breaks = np.linspace(minval ,maxval ,nbreaks)
color = plt.cm.afmhot_r

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

for i in range(0, nyear, dt):
    print(i)
    tt = i

    fig = plt.figure(figsize=(10,8))


    ax = fig.add_subplot(1, 1, 1,projection=proj)
    cs = ax.tricontourf(mesh.x2, mesh.y2, elem2, runoff[tt,:] , extend='both', cmap=color, levels=breaks)
    ax.set_extent([-180,180, -90,90], crs=proj)
    ax.set_title('runoff   year='+str(tt), fontsize=15)
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
    cbar.set_label(label='runoff (m/s)', size=13, ) #weight='bold')
    cbar.ax.tick_params(labelsize=15)  #labelsize=15
    # cbar.set_ticks(np.arange(minval,maxval+0.1,0.25))

    fig.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.15,right=0.94,top=0.98, wspace=0.2,hspace=0.1)

    figname =f'plots/plots_{i+1:04d}.png'
    plt.savefig(figname,bbox_inches='tight',)
    plt.close()

    image = imageio.imread(figname)
    frames.append(image)


gifname = "plots.gif"                    #
# Save to GIF
imageio.mimsave(gifname, frames, duration=0.3)  # duration in seconds per frame


























#-
