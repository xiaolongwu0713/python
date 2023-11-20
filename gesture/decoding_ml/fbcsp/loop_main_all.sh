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
for sid in 11 12
do
#sid='41'
  echo "Start sid: $sid"
    fs=1000
    if [ $sid -eq 11 ] || [ $sid -eq 12 ];then
      fs=500
    fi
     #python main_gen_data.py $sid $network fs wind stride
     python main.py $sid $fs
     echo "Training finish for sid: $sid"
done
#done

