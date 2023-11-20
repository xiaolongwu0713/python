#!/bin/bash

for sid in  1 2 3 4 5 6 7 8 9 10
do
  /cygdrive/c/ProgramData/Anaconda3/envs/bci2/python.exe channel_selection.py $sid
done

