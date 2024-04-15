#!/bin/bash
cv=1
for sid in 2 4 10 13 17 29 32 41
do
  sid=10 # choose a single user to test

  for cv in 1 2 3 4 5
  do
    /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe main_gen.py $sid $cv

    break # test for a single user only
  done
  break
done
#done

