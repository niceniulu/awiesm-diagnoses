#!/usr/bin/env  python3


import argparse, os, sys
#sys.path.append("../")

from netCDF4 import Dataset, MFDataset
import pyfesom2 as pf
import numpy as np
#from mpl_toolkits.basemap import Basemap
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm

def read_data(finname, var, transpose=True):
    with Dataset(finname,'r') as fin:
        data = fin.variables[var]
        ndim = len(data.dimensions)

        # Check if time dimension is there.
        if "time" in data.dimensions:
            print("data dimensions:", data.dimensions, data.shape)
        else:
            raise ValueError('No time dimension!')

        if transpose:
            if ndim == 3:
                if (
                    ("nz1" in data.dimensions)
                    and ("nod2" in data.dimensions)
                ):
                    if (data.dimensions != ("time", "nod2", "nz1")):
                        #data = data.transpose("time", "nod2", "nz1")
                        data = np.transpose(data, axes=(0,2,1))

                elif (
                    ("nz" in data.dimensions)
                    and ("nod2" in data.dimensions)
                ):
                    if (data.dimensions != ("time", "nod2", "nz")):
                        #data = data.transpose("time", "nod2", "nz")
                        data = np.transpose(data, axes=(0,2,1))

                elif (
                    ("nz1" in data.dimensions)
                    and ("elem" in data.dimensions)
                ):
                    if (data.dimensions != ("time", "elem", "nz1")):
                        #data = data.transpose("time", "elem", "nz1")
                        data = np.transpose(data, axes=(0,2,1))
                elif (
                    ("nz" in data.dimensions)
                    and ("elem" in data.dimensions)
                ):
                    if  (data.dimensions != ("time", "elem", "nz")):
                        #data = data.transpose("time", "elem", "nz")
                        data = np.transpose(data, axes=(0,2,1))
                else:
                    raise ValueError('dimension is unknown!')

            # elif ndim == 2:
            #     if (
            #         ("nz1" in data.dimensions)
            #         and ("nod2" in data.dimensions)
            #         and (data.dimensions != ("nod2", "nz1"))
            #     ):
            #         #data = data.transpose("nod2", "nz1")
            #         data = np.transpose(data)
            #     elif (
            #         ("nz" in data.dimensions)
            #         and ("nod2" in data.dimensions)
            #         and (data.dimensions != ("nod2", "nz"))
            #     ):
            #         #data = data.transpose("nod2", "nz")
            #         data = np.transpose(data)
            #
            #     elif (
            #         ("nz1" in data.dimensions)
            #         and ("elem" in data.dimensions)
            #         and (data.dimensions != ("elem", "nz1"))
            #     ):
            #         data = np.transpose(data)
            #     elif (
            #         ("nz" in data.dimensions)
            #         and ("elem" in data.dimensions)
            #         and (data.dimensions != ("elem", "nz"))
            #     ):
            #         data = np.transpose(data)
            #     else:
            #         raise ValueError('dimension is unknown!')

        datanew = data[:]

    return datanew, ndim





def main(meshpath, finname,
         foutname = 'data_regular.nc',
         depth_range=None,
         varnames = ['temp', 'salt'],
         nlon = 1440,
         nlat = 720,
         abg = [0.,0.,0.]):

    mesh = pf.load_mesh(meshpath,abg=abg, usepickle=False)


    # get the depth information
    if depth_range is None:
        depths = mesh.zlev[:-1]
    else:
        idx = (np.abs(mesh.zlev) >= depth_range[0]) & (np.abs(mesh.zlev) <= depth_range[1])
        depths = mesh.zlev[idx]

    nlev = len(depths)

    # the new regular grids:
    lon = np.linspace(-180, 180, nlon)
    lat = np.linspace(-90, 90, nlat)
    lons, lats = np.meshgrid(lon,lat)


    # read the time dim:
    fin = Dataset(finname,'r')
    timein = fin.variables['time']
    ntime = len(timein)

    # write data
    with Dataset(foutname, 'w') as fout:
        fout.createDimension('lat', nlat)
        latout = fout.createVariable('lat',np.float32, ('lat',))
        latout.units = 'degree_north'
        latout.standard_name = 'latitude'
        latout[:] = lat

        fout.createDimension('lon', nlon)
        lonout = fout.createVariable('lon',np.float32,('lon',))
        lonout.units = 'degree_east'
        lonout.standard_name = 'longitude'
        lonout[:] = lon

        fout.createDimension('time', None)
        timeout = fout.createVariable('time',np.float32,('time',))
        timeout.setncatts(timein.__dict__)
        timeout[:] = timein[:]

        fout.createDimension('level', None)
        levout = fout.createVariable('level',np.float32,('level',))
        levout.units = 'm'


        fout.description =  'interpolated fesom2.0 data'

        for thevar in varnames:
            print(">>>>>>>>>> processing >>> ",thevar)
            # read data
            data, ndim = read_data(finname, thevar)
            print(data.shape)

            varout = fout.createVariable(thevar,np.float32,('time','level','lat','lon'))

            for it in range(0,ntime):
                print("> time step ", it)
                if ndim == 3:
                    levout[:] = depths
                    for iz in range(0,nlev):
                        ilev = pf.ind_for_depth(depths[iz], mesh)
                        print("> level:", depths[iz])
                        level_data = data[it, :, ilev]
                        level_data[level_data==0] = np.nan
                        #idist_fast = pf.fesom2regular(level_data, mesh, lons, lats, how='idist', k=20)
                        idist_fast = pf.fesom2regular(level_data, mesh, lons, lats )
                        # idist_fast = np.where(idist_fast.mask, 0., idist_fast) ####  set masked values to 0., for interpolation later
                        varout[it,iz,:,:] = idist_fast

                elif ndim == 2:
                    levout[:] = 0. # meter
                    print("> surface")
                    level_data = data[it, :]
                    idist_fast = pf.fesom2regular(level_data, mesh, lons, lats )
                    # idist_fast = np.where(idist_fast.mask, 0., idist_fast) ####  set masked values to 0., for interpolation later
                    varout[it,0,:,:] = idist_fast



    # close input data
    fin.close()
    return


# #### defination:

# meshpath = '/home/a/a270075/ba0989/pool/meshes/38k/mesh_GLAC38_PM38/'
meshpath = '/home/a/a270075/ba0989/pool/meshes/base21k/mesh_ice38pma1_ocean21_shallowMed/'

finname = 'runoff_orig.nc'  #salt.nc'
foutname = 'runoff_reg.nc'  #salt_reg.nc'
depth_range = None
varnames = ['runoff', ]


main(meshpath, finname,
         foutname = foutname,
         depth_range=depth_range,
         varnames = varnames,
         nlon = 1440,
         nlat = 720,
         abg=[0.,0.,0.],
         )


# parser = argparse.ArgumentParser(description='Input options')
# parser.add_argument("meshpath", type=str, help='FESOM mesh path')
# parser.add_argument("datain", type=str, help='FESOM data path')
# parser.add_argument('vars', nargs='*', type=str, help='fesom variables, e.g. sst temp salt')
#
# parser.add_argument('--dataout', default='data_regular.nc', type=str, help='define output data name')
# parser.add_argument('--depthrange',  default=None,  type=int, nargs=2, help='maximum depth; default:None')
#
#
#
#
# arguments = parser.parse_args()
# print(arguments)
#
# meshpath = arguments.meshpath
# finname = arguments.datain
# varnames = arguments.vars
#
# foutname = arguments.dataout
#
# # choose depths: if None, then all the depths
# depth_range = arguments.depthrange
