#!/bin/bash
source ~/.bash_profile
cd /cygdrive/d/tmp/tts/VCTK_British_English_Males

#input="/cygdrive/c/Users/xiaowu/mydriver/python/dSPEECH/pseudowords/sentences_list.txt"
input="/home/xiaowu/mydriver/python/dSPEECH/pseudowords/sentences_list.txt"
i=1

while IFS= read -r line
do
  echo "$line"
  outfile="$i.wav"
  /cygdrive/c/Users/xiaowu/anaconda3/envs/bci/Scripts/tts.exe --text "$line" --out_path $outfile --model_path checkpoint_85000.pth --config_path config.json --speakers_file_path vctk_speakers.json --speaker_idx VCTK_p226 --language_idx en
  i=$(( $i + 1 ))
done < "$input"

