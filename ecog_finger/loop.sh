#/bin/bash
cwd=`pwd`
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/wuxiaolong/venv/Scripts/activate
	echo "workstation"
fi


if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	echo "longsMac"
fi




for sid in 1 2 3 4 5 6 7 8 9
do
  for wind in 100 200 300 400 500 600
  do
    for stride in 10 20 30 40 50 100
    do
      python main_base.py $sid $wind $stride
    done
  done
done
