#!/bin/bash


expid=tran-17k-hosing

datapath=/work/ba0989/a270075/awiesm2.5/experiments-2/timeslices2/${expid}/outdata/fesom/


# cdo cat  ${datapath}/temp.fesom.????.nc  temp.fesom.orig.nc


### select level range and get mean
# cdo -vertmean -sellevel,300/800  temp.fesom.orig.nc   temp.fesom.orig.level.nc



################################################################## salt 
cdo cat  ${datapath}/salt.fesom.????.nc  salt.fesom.orig.nc
cdo -vertmean -sellevel,300/800  salt.fesom.orig.nc   salt.fesom.orig.level.nc
