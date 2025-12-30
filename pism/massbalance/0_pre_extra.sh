#!/bin/bash


expid=tran-17k-hosing
datapath=/home/a/a270075/ba0989/awiesm2.5/experiments-2/timeslices2/${expid}/


cdo  -select,name=tendency_of_ice_amount_due_to_basal_mass_flux,tendency_of_ice_amount_due_to_discharge,tendency_of_ice_amount_due_to_surface_mass_flux,tendency_of_ice_amount_due_to_calving  \
    ${datapath}/outdata/pism/${expid}_pismr_extra_*.ymonmean.nc   ${expid}_extra.nc
