#!/bin/bash
cv=1
for sid in 2 4 10 13 17 29 32 41
do

  for cv in 1 2 3 4 5
  do
    sid=10 # choose a single user
    /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe main_train.py --sid $sid --cv $cv #--load_path 'D:/tmp/python/gesture/DA/cTGAN/sid10/2024_04_10_16_09_06/Model/checkpoint_290.pth'

    break # test for a single user only

  done
  break
done
#done

