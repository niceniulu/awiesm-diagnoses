#!/bin/bash


expid=tran-17k-hosing

datapath=/work/ba0989/a270075/awiesm2.5/experiments-2/timeslices2/${expid}/outdata/fesom/


# cdo cat  ${datapath}/a_ice.fesom.????01.01.nc  a_ice.fesom.orig.nc
cdo cat  ${datapath}/a_ice.fesom.????.nc  a_ice.fesom.orig.nc
# cdo -yearmean  a_ice.fesom.orig.nc   a_ice.fesom.orig.yearmean.nc



######## one year
ystart=3900
yend=3910
cdo -ymonmean -seldate,${ystart}-01-01,${yend}-12-31   a_ice.fesom.orig.nc  a_ice_oneyear.nc
