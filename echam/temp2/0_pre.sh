#!/bin/bash

function post_echam(){
    # cdo  -select,name=var167,var169,var142,var143,var210     ${echampath}/${expid}_245???.01_echam  echam.grb
    cdo  -select,name=var167,var169     ${echampath}/*_*.01_echam  echam.grb
    cdo  -f nc -t echam6 copy  echam.grb  echam.nc
    #
    # temp2 2d climatology
    cdo  -yearmean -selmon,6,7,8     echam.nc    echam_temp2_summer.nc
    cdo  -yearmean -selmon,1,11,12     echam.nc    echam_temp2_winter.nc
    cdo  -yearmean   echam.nc    echam_temp2_annual.nc
    rm echam.grb
}



# ystart=2300
# yend=2900
# var=temp2
echampath=../../../outdata/echam/

post_echam
