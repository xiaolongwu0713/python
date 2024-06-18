#!/bin/bash

channel_to_select=10
cv=1
for sid in 2 4 10 13 17 29 32 41
do

  for cv in 1 2 3 4 5
  do
    sid=10 # choose a single user
    /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe selection_gumbel.py $sid $channel_to_select $cv
    break # test for a single user only

  done
  break
done
#done
