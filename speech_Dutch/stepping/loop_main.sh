#!/bin/bash
mel_bins=23
for sid in 1 2 3 4 5 6 7 8 9 10
#for sid in 1 3
do
  /cygdrive/c/ProgramData/Anaconda3/envs/bci2/python.exe main_stepping.py $sid
done


