#!/bin/bash
# usage: ./loop_main_gen_data.sh gen_data/retrain
# default window and stride: ./loop_main_gen_data.sh 500 100
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/Public/venv/bci/Scripts/activate
	sidsfile="/home/wuxiaolong/mydrive/meta/gesture/info/Info.txt"
	good_sidsfile="/home/wuxiaolong/mydrive/meta/gesture/good_sids.txt"
	echo "workstation"
fi

if [[ $HOSTNAME == "DESKTOP-NP9A9VI"  ]];then
	source /cygdrive/c/Users/Public/venvs/bci/Scripts/activate
	sidsfile="/home/xiaol/mydrive/meta/gesture/info/Info.txt"
	good_sidsfile="/home/xiaol/mydrive/meta/gesture/good_sids.txt"
	echo "DESKTOP"
fi

if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	sidsfile="/Users/long/mydrive/meta/gesture/info/Info.txt"
	good_sidsfile="/Users/long/mydrive/meta/gesture/good_sids.txt"
	echo "longsMac"
fi

#declare -a sids
sids=()
while IFS= read -r line
do
 sid=${line%,*}
 sids+=("$sid")
 #echo $sid
 #echo ${sids[@]}
done < "$sidsfile"
#echo ${sids[@]}

good_sids=()
while IFS= read -r line
do
 sid=${line%}
 good_sids+=("$sid")
 #echo $sid
 #echo ${sids[@]}
done < "$good_sidsfile"

#echo ${good_sids[@]}

for sid in ${sids[@]} #${good_sids[@]} #${sids[@]}
#for sid in 25 29 32 34 41
do
  sid=10
  time1=$(date +%s)
  echo "************* Method:$1; Sid:$sid; *************"
  # python main_gen_data.py $sid fs wind stride task(gen_data/retrain) gen_methods(VAE/gan/wgan)
  # call from CMD: loop_main_gen_data.sh VAE/DCGAN/WGANGP($1) resumt/fresh(resume or a fresh training)($2) epcch_number($3)
  # Resume example: ./loop_main_gen_data.sh wgan_gp resume 50: train for another 50 epochs
  # fresh training example: ./loop_main_gen_dta.sh wgan_gp fresh 200
  if [ $2 = 'resume' ]; then
    echo "Resume training of data augmentation for sid:$sid using $1."
  elif [ $2 = 'fresh' ]; then
    echo "Training data augmentation for sid:$sid using $1."
  else
    echo "Mush be either 'resume' or 'fresh'."
    break
  fi

  python main_gen_data.py --sid $sid --fs 1000 --wind 500 --stride 200 --gen_method $1 --continuous $2 --epochs $3 --selected_channels 'Yes'

  #time2=$(date +%s)
  #duration=$(( time2 - time1 ))
  #echo "Training on sid:$sid done, using $(( duration / 3600 )) hours."

  break # test and break
done
#done

