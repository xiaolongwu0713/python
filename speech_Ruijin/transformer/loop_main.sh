#!/bin/bash
source ~/.bash_profile
time_stamp=`date +"%Y_%m_%d_%H_%M"`
#time_stamp='2023_09_03_12_05'
dataname='SingleWordProductionDutch' #'mydata'/'SingleWordProductionDutch'

if [ $hostname = "Yoga" ]
then
  for sid in  3 #1 2 4 5 6 7 8 9 10 #3
  do
    echo "-------  Training ---------"
    /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe main.py $sid $dataname $time_stamp
    echo "-------  inference --------"
    /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe inference.py $sid $dataname $time_stamp
    echo "------- synthesize --------"
    /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe synthesize_waveglow.py $sid $dataname $time_stamp
  done

else
  for sid in  1 2 4 5 6 7 8 9 10 #3
  do
    echo "-------  Training ---------"
    /cygdrive/c/ProgramData/Anaconda3/envs/bci2/python.exe main.py $sid $dataname $time_stamp
    echo "-------  inference --------"
    /cygdrive/c/ProgramData/Anaconda3/envs/bci2/python.exe inference.py $sid $dataname $time_stamp
    echo "------- synthesize --------"
    /cygdrive/c/ProgramData/Anaconda3/envs/bci2/python.exe synthesize_waveglow.py $sid $dataname $time_stamp
  done
fi



