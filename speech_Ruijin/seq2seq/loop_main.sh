#!/bin/bash

source ~/.bash_profile
time_stamp=`date +"%Y_%m_%d_%H_%M"`
#time_stamp='2023_09_03_12_05'
dataname='SingleWordProductionDutch' #'mydata'/'SingleWordProductionDutch'
mel_bins=23

for sid in 3 # 1 2 3 4 5 6 7 8 9 10
#for sid in 1 3
do
  echo "-------  Training on $hostname ---------"
  /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe main_seq2seq_fixed_window.py $sid $dataname $time_stamp $mel_bins
  echo "-------  inference on $hostname --------"
  /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe inference.py $sid $dataname $time_stamp $mel_bins
done


