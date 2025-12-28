#!/bin/bash


expid=tran-17k-hosing

datapath=/work/ba0989/a270075/awiesm2.5/experiments-2/timeslices2/${expid}/outdata/fesom/


# cdo cat  ${datapath}/sss.fesom.????01.01.nc  sss.fesom.orig.nc
cdo cat  ${datapath}/sss.fesom.????.nc  sss.fesom.orig.nc
cdo -yearmean  sss.fesom.orig.nc   sss.fesom.orig.yearmean.nc
