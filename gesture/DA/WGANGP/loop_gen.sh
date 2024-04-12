#!/bin/bash
# usage: ./loop_main_gen_data.sh gen_data/retrain
# default window and stride: ./loop_main_gen_data.sh 500 100

for sid in 2 4 10 13 17 29 32 41
#for sid in 25 29 32 34 41
do
  sid=10
  time1=$(date +%s)
  echo "************* Method:$1; Sid:$sid; *************"
  # python main_gen_data.py $sid fs wind stride task(gen_data/retrain) gen_methods(VAE/gan/wgan)
  # call from CMD: loop_main_gen_data.sh VAE/DCGAN/WGANGP($1) resumt/fresh(resume or a fresh training)($2) epcch_number($3)
  # Resume example: ./loop_main_gen_data.sh wgan_gp resume 50: train for another 50 epochs
  # fresh training example: ./loop_main_gen_dta.sh wgan_gp fresh 200

  /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/python.exe main_gen_data.py

  #time2=$(date +%s)
  #duration=$(( time2 - time1 ))
  #echo "Training on sid:$sid done, using $(( duration / 3600 )) hours."

  break # test and break
done
#done

