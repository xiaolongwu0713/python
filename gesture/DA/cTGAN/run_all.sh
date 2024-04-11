#!/bin/bash

for sid in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 171 18 19 20 21 22 23 24 25 26 27 28 29 30
do
  sid=10 # choose a single user to test
  /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe main.py --sid $sid

  break # test for a single user only
done
#done

