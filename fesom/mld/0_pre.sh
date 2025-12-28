#!/bin/bash


expid=tran-17k-hosing

datapath=/work/ba0989/a270075/awiesm2.5/experiments-2/timeslices2/${expid}/outdata/fesom/


# cdo cat  ${datapath}/MLD1.fesom.????01.01.nc  MLD1.fesom.orig.nc
cdo cat  ${datapath}/MLD1.fesom.????.nc  MLD1.fesom.orig.nc
cdo -yearmean  MLD1.fesom.orig.nc   MLD1.fesom.orig.yearmean.nc
