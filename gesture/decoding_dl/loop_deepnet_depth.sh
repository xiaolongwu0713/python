#!/bin/bash
# usage: ./loop_main_gen_data.sh wind=500 stride=50
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/wuxiaolong/venv/Scripts/activate
	echo "workstation"
fi


if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	echo "longsMac"
fi

inputfile="H:/Long/data/gesture/preprocessing/Info.txt"
#declare -a sids
sids=()
while IFS= read -r line
do
 sid=${line%,*}
 sids+=("$sid")
 #echo $sid
 #echo ${sids[@]}
done < "$inputfile"
#echo ${sids[@]}

#for sid in ${sids[@]}
#for sid in 4 10 13 41
#do
sid='41'
  echo "Start sid: $sid"
  network='deepnet_changeDepth'
  #for network in 'eegnet'# 'shallowFBCSPnet' 'deepnet' 'resnet'
  #do
  for depth in  1 2 3 4 5 6
  do
     echo "*************Sid $sid on $network*************"
     #python main_gen_data.py $sid $network fs wind stride
     python main_all.py $sid $network 1000 $1 $2 $depth
     echo "Training finish for sid: $sid on $network" 
  done
#done

