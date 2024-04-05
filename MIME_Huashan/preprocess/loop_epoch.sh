#!/bin/bash
source ~/.bash_profile
time_stamp=`date +"%Y_%m_%d_%H_%M"`
#time_stamp='2023_09_03_12_05'
scaler='std'

if [ $hostname = "Yoga" ]
then
  for sid in  3 4 5 6 # 1 2
  do
    echo "-------  Epoching Sid: $sid ---------"
    /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe epoch.py $sid $scaler $time_stamp

  done
else
  echo "Hostname not right."
fi



