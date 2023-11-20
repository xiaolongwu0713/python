#!/bin/bash
# usage: ./loop_main_gen_data.sh wind stride
# ./loop_main_gen_data.sh 500 50 | ./loop_main_gen_data.sh 200 50
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
for wind in 500 400 300 200 100
do
for sid in 4 10 13 29 41 #='29'
do
  echo "Start sid: $sid"
  #network='deepnet_da'
  #for network in 'eegnet' 'shallowFBCSPnet' 'deepnet' 'resnet' 'deepnet_da'
  #do
  network='resnet'
     echo "*************Sid $sid on $network*************"
     #python main_gen_data.py $sid $network fs wind stride
     python main_all.py $sid $network 1000 $wind 50
     echo "Training finish for sid: $sid on $network" 
  #done
done
done

