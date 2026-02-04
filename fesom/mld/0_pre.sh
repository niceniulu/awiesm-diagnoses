#!/bin/bash


# expid=tran-24k-hosing

datapath=../../../outdata/fesom/


# cdo cat  ${datapath}/MLD1.fesom.????01.01.nc  MLD1.fesom.orig.nc
cdo cat  ${datapath}/MLD1.fesom.????.nc  MLD1.fesom.orig.nc
cdo -yearmean  MLD1.fesom.orig.nc   MLD1.fesom.orig.yearmean.nc
