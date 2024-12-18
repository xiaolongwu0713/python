#!/bin/bash

# possible network: 'deepnet'/'resnet'/'EEGnet'
# possible train_mode: 'original'/'selected_channels'/'DA'
# possible method of selected_channels: 'gumbel'/'stg'/'mannual'
# possible method of DA: 'VAE'/'cTGAN'/'WGANGP'/'NI'
for sid in 25 29 32 34 41
do
  sid=41 # choose a single user to test

  for cv in 1 2 3 4 5
  do

    network='deepnet'
    train_mode='original' #'original'

    if [ $train_mode = 'original' ]
    then
      echo "CMD: Training sid: $sid using $network."
      /cygdrive/d/Users/xiaowu/anaconda3/envs/bci/python.exe main_all.py $sid $network $train_mode $cv
    elif [ $train_mode = 'DA' ]
    then
      DA_method='CWGANGP' #'cTGAN'
      /cygdrive/d/Users/xiaowu/anaconda3/envs/bci/python.exe main_all.py $sid $network $train_mode $DA_method $cv
    fi

    break # test for a single user only
  done

  break
done
#done

