#!/bin/bash


expid=tran20-24k
datapath=/home/a/a270075/ba0989/awiesm2.5/experiments-2/${expid}/


cdo  -select,name=thk,topg  ${datapath}/restart/pism/${expid}_pismr_restart_*.nc  ${expid}_thk_topg.nc
