#!/bin/bash

imdir=$1
codedir=$2
count=$3
params=$4

imdir_name=$(echo $1 | rev | cut -f1 -d/ | rev)

code="${codedir}/newPanoramaStitcher"
out=""
out2=""
j=0
k=0
warp="spherical"
matcher="affine"

for i in $( ls -1 ${imdir} )
do
    line="$(echo ${i})"
    ext=$(echo $line | cut -f2 -d.);
    if [ $ext == "jpg" ] || [ $ext == "JPG" ] || [ $ext == "jpeg" ] || [ $ext == "JPEG" ] || [ $ext == "png" ] || [ $ext == "PNG" ];then
	j=$(( $j + 1 ))
	out="${out}${imdir}/${line} "
    fi
    if [ $count == "$j" ];then
	args="${out} --try_cuda no --features akaze --output ${imdir_name}_$k.jpg --work_megapix 1 --match_conf 0.1 --conf_thresh 0.1 --matcher ${matcher} --match_type adjacent --warp ${warp}"
	out2="${out2} ${imdir_name}_$k.jpg"
	#args="${out} --d3 --mode panorama --output ${imdir_name}_$k.jpg"
	echo $args
	#if [ $k == "2" ];then
	$code $args
	#fi
	#out="${imdir}/${line} "
	out=""
	j=0
	k=$(( $k + 1 ))
    fi
done
args="${out} --try_cuda no --features akaze --output ${imdir_name}_$k.jpg --work_megapix 1 --match_conf 0.1 --conf_thresh 0.1 --matcher ${matcher} --match_type adjacent --warp ${warp}"
out2="${out2} ${imdir_name}_$k.jpg"
echo $args
$code $args

args="${out2} --try_cuda no --features akaze --output ${imdir_name}_final.jpg --work_megapix 0.8 --match_conf 0.1 --conf_thresh 0.05 --matcher ${matcher} --blend feather --match_type adjacent --warp ${warp}"
echo $args
$code $args
