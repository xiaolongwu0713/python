#!/bin/bash
cwd=`pwd`
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/wuxiaolong/venv/Scripts/activate
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

for selction_lr in 0.0001 0.0003 0.001 0.003 0.007 0.01 0.03 0.05
do
  for network_lr in 0.001 0.007 0.01 0.03 0.05 0.07 0.1 
  do
      python selection_gumbel.py $selction_lr $network_lr
  done
done
