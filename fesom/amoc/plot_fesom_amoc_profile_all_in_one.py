#!/usr/bin/env  python3
# coding: utf-8

import pyfesom2 as pf
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import netCDF4 as nc
import argparse, os, sys

def  amoc_depth_onetime(meshdir, datadir, ystart, yend ,
                        foutname='amoc_depth.nc',
                        nlats=361,
                        choose_mask = "Atlantic_MOC",
                        **kwargs):
    ''' masks:
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

    #kwargs = {'use_cftime': 'True', }
    # get data
    data1 = pf.get_data(datadir, 'w', range(ystart,yend), mesh, how="mean", compute=True, **kwargs )
    data2 = pf.get_data(datadir, 'bolus_w', range(ystart,yend), mesh, how="mean", compute=True , **kwargs )
    print("--> years, nodes, levels", data1.shape)
    data=data1+data2
    print(data.shape)

    # get mask
    # mask1 = pf.get_mask(mesh, "Atlantic_Basin")
    # mask2 = pf.get_mask(mesh, "Arctic_Basin")
    # mask3 = mask1|mask2

    lats, moc = pf.xmoc_data(mesh, data, mask=choose_mask, nlats=nlats)
    print(moc.shape)

    # write to file
    fout = nc.Dataset(foutname, 'w')

    fout.createDimension('lat', nlats)
    latout = fout.createVariable('lat',np.float32, ('lat',))
    latout.units = 'degree_north'
    latout.standard_name = 'latitude'
    latout[:] = lats

    fout.createDimension('level', len(mesh.zlev[:]))
    levout = fout.createVariable('level',np.float32,('level',))
    levout.units = 'm'
    levout.positive = "down"
    levout.axis = "Z"
    levout[:] = np.abs(mesh.zlev[:])


    fout.createDimension('time', None)
    timeout = fout.createVariable('time',np.float32,('time',))
    timeout.units = 'years since '+str(int((ystart+yend)/2))+'-01-01'

    varout = fout.createVariable('AMOC',np.float32,('time','level','lat'))
    varout[0,:,:] = np.transpose(moc)

    fout.close()

    # plot

    plt.figure(figsize=(10, 3))
    pf.plot_xyz(mesh, moc, xvals=lats, maxdepth=7000, cmap=cm.seismic, levels = np.linspace(-20, 20, 41),
                facecolor='gray', label='Sv')
    plt.savefig(foutname.replace('.nc','.png'))
    plt.show(block=False)
    plt.pause(3)
    plt.close()


    return






if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Input options')
    # parser.add_argument("-meshpath", type=str, help='FESOM mesh path')
    # parser.add_argument("-datapath", type=str, help='FESOM data path')
    # parser.add_argument('-ystart',  type=int, help='start year')
    # parser.add_argument('-yend',  type=int, help='end year')
    # parser.add_argument('-outdir',  type=str, help='output directory')
    # parser.add_argument('-expname', default='EXP', type=str, help='define name for plot')
    #
    # parser.add_argument('--meshalpha',  default=0., type=float, help='mesh alpha')
    # parser.add_argument('--meshbeta',  default=0., type=float, help='mesh beta')
    # parser.add_argument('--meshgamma',  default=0., type=float, help='mesh gamma')
    #
    #
    #
    # arguments = parser.parse_args()
    # print(arguments)

    expname='tran-17k-hosing'
    meshpath='/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'
    datapath='/home/a/a270075/ba0989/awiesm2.5/experiments-2/timeslices2/tran-17k-hosing/outdata/fesom/'
    ystart = 3900 #2000
    yend = 3915


    # kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}01.01.nc"}
    kwargs = {'use_cftime': 'True', 'naming_convention' : "custom", 'naming_template':"{}.{}.{}.nc"}

    print()

    ### set the mesh and data information
    print("- the mesh information")
    fname = 'amoc_depth_'+str(ystart)+'-'+str(yend)+'-'+ expname+'.nc'
    amoc_depth_onetime(meshpath,datapath, ystart, yend,
                        foutname=fname,nlats=361, **kwargs)
