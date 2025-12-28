#!/usr/bin/env  python3
# coding: utf-8


import pyfesom2 as pf
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import netCDF4 as nc
import argparse, os, sys


def  amoc_timeseries(meshdir, datadir, ystart, yend ,
                    whichlat=[26.5, 35, 40],
                    foutname='amoc_ts.nc',
                    choose_mask = "Atlantic_MOC",
                    **kwargs):
    '''
    masks:
    Ocean Basins:
        "Atlantic_Basin"
        "Pacific_Basin"
        "Indian_Basin"
        "Arctic_Basin"
        "Southern_Ocean_Basin"
        "Mediterranean_Basin"
        "Global Ocean"
        "Global Ocean 65N to 65S"
        "Global Ocean 15S to 15N"
    MOC Basins:
        "Atlantic_MOC"
        "IndoPacific_MOC"
        "Pacific_MOC"
        "Indian_MOC"
    Nino Regions:
        "Nino 3.4"
        "Nino 3"
        "Nino 4"
    '''



    mesh = pf.load_mesh(meshdir)

    #kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}01.01.nc"}
    # get data
    data1 = pf.get_data(datadir, 'w', range(ystart,yend), mesh, how="ori", compute=False, **kwargs )
    data2 = pf.get_data(datadir, 'bolus_w', range(ystart,yend), mesh, how="ori", compute=False , **kwargs)
    print("--> years, nodes, levels", data1.shape)
    data=data1+data2

    # get mask
    # mask1 = pf.get_mask(mesh, "Atlantic_Basin")
    # mask2 = pf.get_mask(mesh, "Arctic_Basin")
    # mask3 = mask1|mask2

    years = range(ystart, yend)

    # get moc
    moc = []
    print('Calculate MOC ...')
    for i in range(data.shape[0]):
        print('   ', data.time[i].values)
        lats, moc_time = pf.xmoc_data(mesh, data[i,:,:], mask=choose_mask, nlats=361)
        moc.append(moc_time)


    # write to file
    fout = nc.Dataset(foutname, 'w',format='NETCDF4_CLASSIC')
    fout.createDimension('time', None)
    timeout = fout.createVariable('time',np.float32,('time',))
    # timeout.units = 'years since '+str(data.time[0].values)
    # timeout[:] = years  #np.arange(1,len(moc)+1)
    timeout.units = 'years since 0001-01-01 '
    timeout[:] = years  #np.arange(1,len(moc)+1)


    # get index
    for j in whichlat:
        indlat = np.where(lats == j)
        print('--> index_lat:', indlat[0] ,'lat:', lats[indlat[0]])
        moc_lat = []
        for i in range(len(moc)):
            moc_lat.append(moc[i][indlat,:].max())

        varname = 'AMOC_'+str(lats[indlat[0][0]])
        varout = fout.createVariable(varname,np.float32,('time',))
        varout[:] = moc_lat[:]

        plt.plot(years, moc_lat, label=j)

    fout.close()

    # plot
    plt.legend()
    #plt.ylim(5,25)
    plt.xlabel('Years')
    plt.ylabel('AMOC (Sv)')
    plt.savefig(foutname.replace('.nc','.png'))
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return



if __name__ == '__main__':


    # kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}01.01.nc"}
    kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}.nc"}

    # expname='tran-38pma1-v2-38k'
    meshpath='/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'
    datapath='/home/a/a270075/ba0989/awiesm2.5/experiments-2/timeslices2/tran-17k-hosing/outdata/fesom/'
    ystart = 3000 #2000
    yend = 3915 



    print()

    ### set the mesh and data information
    print("- the mesh information")
    # fname = 'amoc_timeseries_'+str(ystart)+'-'+str(yend)+'-'+ expname+'.nc'
    fname = 'amoc_timeseries.nc'

    amoc_timeseries(meshpath,datapath, ystart, yend,
                    whichlat=[26.5, 35, 40], foutname=fname, **kwargs)
