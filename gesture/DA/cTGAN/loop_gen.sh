#!/bin/bash
cv=1
for sid in 2 4 10 13 17 29 32 41
do
  sid=10 # choose a single user to test
  /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe main.py --sid $sid --cv $cv

  break # test for a single user only
done
#done

