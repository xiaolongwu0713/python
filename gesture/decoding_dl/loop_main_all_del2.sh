#!/bin/bash
# usage: ./loop_main_gen_data.sh
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
  # sid=10 # choose a single user to test

  #network='deepnet_da'
  #for network in 'eegnet' 'shallowFBCSPnet' 'deepnet' 'resnet' 'deepnet_da'
  #do

  network='deepnet'
  # usage: ./loop_main_all.sh 'selected_channels'/'DA'  'gumbel'/'stg'/'mannual' or 'VAE'/'GAN'/'WGAN'

  # vanilla training: call: ./loop_main_all.sh # (no selection, or augmentation), and uncomment below two lines
  echo "Training sid: $sid using $network."
  python main_all.py $sid $network 1000 500 100

  # call: ./loop_main_all.sh 'selected_channels' 'stg', and uncomment below two lines
  #echo "Training sid: $sid using $2 selecting method. "
  #python main_all.py $sid $network 1000 500 200 $1 $2

  # re-train with data augmented from WGAN using 200 epochs
  #call: ./loop_main_all.sh DA WGAN_GP 200;
  #echo "Training finish for sid: $sid using data augmented by $2. "
  #python main_all.py $sid $network 1000 500 200 $1 $2 $3
  #call: ./loop_main_all.sh DA NI
  #python main_all.py $sid $network 1000 500 200 $1 $2 $3

  # break # test for a single user only
done
#done

