#!/bin/bash


expid=tran-17k-hosing

datapath=/work/ba0989/a270075/awiesm2.5/experiments-2/timeslices2/${expid}/outdata/fesom/


# cdo cat  ${datapath}/sst.fesom.????01.01.nc  sst.fesom.orig.nc
cdo cat  ${datapath}/sst.fesom.????.nc  sst.fesom.orig.nc
cdo -yearmean  sst.fesom.orig.nc   sst.fesom.orig.yearmean.nc
