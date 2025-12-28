#!/usr/bin/env  python3
# coding: utf-8


import numpy as np
import netCDF4 as nc
import pyfesom2 as pf
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cmocean


exps = ['amoc_timeseries.nc',
        # 'amoc_timeseries2.nc',
        # 'tran20-32k-sed30',
        # 'tran20-32k-new04',
       ]

datas = [ i  for i in exps[:] ]
datas


################# figure
fig = plt.figure(figsize=(12,5))
fig.subplots_adjust(left=0.05,right=0.82, top=0.95, bottom=0.05, hspace=0.1,wspace=0.05)
ax = fig.add_subplot(1,1,1)

colors = ['red',] *10   #'blue', 'orange']

# years = np.arange(-32000,0+1,20)

for i in  range(0,len(exps)):
    with nc.MFDataset(datas[i], 'r') as ff:
        amoc = ff.variables['AMOC_26.5'][:]
        years = ff.variables['time'][:]

    ax.plot(years[0:len(amoc)], amoc, linewidth=2, ls='-', color=colors[i], label=exps[i])
    ax.legend()
    # ax.set_xlim(38, 13)
    # ax.set_ylim(460, 520)
    ax.set_xlabel("Year ", fontsize=15)
    ax.set_ylabel("AMOC index (Sv)", fontsize=15)
    ax.tick_params(axis='y', **dict(size=3, width=1.5, labelsize=14))
    ax.tick_params(axis='x', **dict(size=3, width=1.5, labelsize=14))

plt.savefig('1_plot_amoc.png',bbox_inches='tight',)
# plt.show()
# plt.close()
