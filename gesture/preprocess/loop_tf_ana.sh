#!/bin/bash
#
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/Public/venv/bci/Scripts/activate
	sidsfile="/home/wuxiaolong/mydrive/meta/gesture/info/Info.txt"
	good_sidsfile="/home/wuxiaolong/mydrive/meta/gesture/good_sids.txt"
	echo "workstation"
fi

if [[ $HOSTNAME == "DESKTOP-NP9A9VI"  ]];then
	source /cygdrive/c/Users/Public/venvs/bci/Scripts/activate
	sidsfile="/home/xiaol/mydrive/meta/gesture/info/Info.txt"
	good_sidsfile="/home/xiaol/mydrive/meta/gesture/good_sids.txt"
	echo "DESKTOP"
fi

if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	sidsfile="/Users/long/mydrive/meta/gesture/info/Info.txt"
	good_sidsfile="/Users/long/mydrive/meta/gesture/good_sids.txt"
	echo "longsMac"
fi

#declare -a sids
sids=()
while IFS= read -r line
do
 sid=${line%,*}
 sids+=("$sid")
 #echo $sid
 #echo ${sids[@]}
done < "$sidsfile"
#echo ${sids[@]}

good_sids=()
while IFS= read -r line
do
 sid=${line%}
 good_sids+=("$sid")
 #echo $sid
 #echo ${sids[@]}
done < "$good_sidsfile"



for sid in ${sids[@]} #${good_sids[@]}
do
  echo "Start sid: $sid"
    #python tf_all_channel.py $sid
    python tf_ersd_slide_f.py $sid
  #done
done

