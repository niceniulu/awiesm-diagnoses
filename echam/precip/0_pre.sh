#!/bin/bash

function post_echam(){
    # cdo  -select,name=var167,var169,var142,var143,var210     ${echampath}/${expid}_245???.01_echam  echam.grb
    cdo  -select,name=var142,var143     ${echampath}/${expid}_*.01_echam  echam.grb
    cdo  -f nc -t echam6 copy  echam.grb  precip.nc
    #
    # precip 2d climatology
    cdo  -yearmean -selmon,6,7,8  -selvar,$var   precip.nc    ${expid}_echam_precip_summer.nc
    cdo  -yearmean -selmon,1,11,12  -selvar,$var   precip.nc    ${expid}_echam_precip_winter.nc
    rm echam.grb
}


expid=tran-17k-hosing

# ystart=2300
# yend=2900
var=aprl,aprc
echampath=/home/a/a270075/ba0989/awiesm2.5/experiments-2/timeslices2/${expid}/outdata/echam/

post_echam
