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

ET="TP004 TP008 TP009 TP012 TP017" # space is necessary  TP003
PD="TP005 TP007 TP010 TP013 TP014 TP015 TP016"
others="TP001 TP002 TP006 TP011 TP018 TP019 TP020"
NC="TN001 TN002 TN003 TN004 TN005 TN006 TN007 TN008 TN009 TN010 TN011 TN012 TN013 TN014 TN015 TN016 TN017 TN018"

target="PD"
testing=0
class_num=2
if [[ $target == "ET" ]]; then
  for sid in ${ET[@]}; do # ${sids[@]}
    if [[ $sid == "TP003" ]]; then
      test_sub=$sid
      val_sub='TP004'
    else
      test_sub=$sid
      val_sub='TP003'
    fi
    date +%Y%m%d%H%M%S
    echo "Test($test_sub); Validata($val_sub)."
    # usage: ./loop_main.sh 'selected_channels'/'DA'  'gumbel'/'stg'/'mannual' or 'VAE'/'GAN'/'WGAN'
    python main_LOO2.py $testing  $class_num $target $test_sub $val_sub
    date +%Y%m%d%H%M%S

    #break # test for a single user only
  done
elif [[ $target == "PD" ]]; then
  for sid in ${PD[@]}; do # ${sids[@]}
    if [[ $sid == "TP005" ]]; then
      test_sub=$sid
      val_sub='TP007'
    else
      test_sub=$sid
      val_sub='TP005'
    fi
    date +%Y%m%d%H%M%S
    echo "Test($test_sub); Validata($val_sub)."
    # usage: ./loop_main.sh 'selected_channels'/'DA'  'gumbel'/'stg'/'mannual' or 'VAE'/'GAN'/'WGAN'
    python main_LOO2.py $testing  $class_num $target $test_sub $val_sub
    date +%Y%m%d%H%M%S

    #break # test for a single user only
  done

  elif [[ $target == "others" ]]; then
  for sid in ${others[@]}; do # ${sids[@]}
    if [[ $sid == "TP001" ]]; then
      test_sub=$sid
      val_sub='TP002'
    else
      test_sub=$sid
      val_sub='TP001'
    fi
    date +%Y%m%d%H%M%S
    echo "Test($test_sub); Validata($val_sub)."
    # usage: ./loop_main.sh 'selected_channels'/'DA'  'gumbel'/'stg'/'mannual' or 'VAE'/'GAN'/'WGAN'
    python main_LOO2.py $testing  $class_num $target $test_sub $val_sub
    date +%Y%m%d%H%M%S

    #break # test for a single user only
  done

fi
