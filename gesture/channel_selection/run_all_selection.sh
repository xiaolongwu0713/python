#!/bin/bash
cwd=`pwd`
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/Public/venv/bci/Scripts/activate
	sidsfile="/home/wuxiaolong/mydrive/meta/gesture/info/Info.txt"
	good_sidsfile="/home/wuxiaolong/mydrive/meta/gesture/good_sids.txt"
	echo "workstation"
fi

if [[ $HOSTNAME == "DESKTOP-NP9A9VI"  ]];then
	source /cygdrive/c/Users/Public/venvs/bci/Scripts/activate
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
done < "$sidsfile"

good_sids=()
while IFS= read -r line
do
 sid=${line%}
 good_sids+=("$sid")
done < "$good_sidsfile"


##########  selection method ################
select_method="stg"  #"gumbel"/"stg"

##############  gumbel ######################
if [[ $select_method == "gumbel"  ]]; then
channel_to_select=10
sid=10
#for sid in ${sids[@]} #2 4 17 25 34  #${sids[@]}
#for sid in  10
for channel_to_select in 11 12 13 14 15
do
  echo $sid
  python selection_gumbel.py $sid $channel_to_select

  ## reproducable testing
  #for repitition in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
  #do
  #  python selection_gumbel.py $sid $channel_to_select $repitition
  #done
done
fi

##############  stg ######################
if [[ $select_method == "stg"  ]];then
for sid in 4 10 13 17 18 25 29 32 34 41 #${sids[@]} #2 4 17 25 34  #${sids[@]}
  do
    for lam in 0.02 0.1 0.2 0.6 1 3
      do
        echo "Perform STG-based selection on SID: $sid. Lambda: $lam."
        python selection_stg.py $sid $lam
      done
  done
fi

