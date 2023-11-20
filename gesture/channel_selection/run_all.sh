#!/bin/bash
cwd=`pwd`
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/Public/venv/bci/Scripts/activate
	inputfile="/home/wuxiaolong/mydrive/meta/gesture/info/Info.txt"
	echo "workstation"
fi

if [[ $HOSTNAME == "DESKTOP-NP9A9VI"  ]];then
	source /cygdrive/c/Users/Public/venvs/bci/Scripts/activate
	echo "DESKTOP"
fi

if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	echo "longsMac"
fi


#inputfile="C:/Users/xiaol/My Drive/meta/gesture/info/Info.txt"
#declare -a sids
sids=()
while IFS= read -r line
do
 sid=${line%,*}
 sids+=("$sid")
 #echo ${sids[@]}
done < "$inputfile"

#for sid in ${sids[@]}
channel_to_select=10
for sid in  2 3 4 10 13 17 18 29 32 41
do
  echo $sid
  #python selection_gumbel.py $sid $channel_to_select
  python test_gumbel.py $sid $channel_to_select
  #python selection_stg.py $sid
done
