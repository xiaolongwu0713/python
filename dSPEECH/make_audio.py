from pdb import set_trace as db

import numpy as np
import librosa
import pydub
from librosa import load
from librosa.util import fix_length
import os
import copy
from pdb import set_trace as db

import soundfile

import librosa.display

# We'll need IPython.display's Audio widget
from IPython.display import Audio

mixdown_order = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,11,12,16,17,18,19,20,21,22,100,24,25,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,74,75,77,78,79,80,81,82,83,84,85,86,23,88,89,90,91,92,93,94,95,96,97,98,87,99,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,100,100,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61,61,62,62,63,63,64,64,65,65,66,66,67,67,68,68,69,69,70,70,71,71,72,72,73,73,73,74,74,75,75,76,76,77,77,78,78,79,79,80,80,81,81,82,82,83,83,84,84,85,85,86,86,23,23,88,88,89,89,90,90,91,91,92,92,93,93,94,94,95,95,96,96,97,97,98,98,87,87,99,99]
wav = 'D:/data/speech_Southmead/audio/mixdown_aligned_noclip.wav'#os.path.join(os.getcwd(),'mixdown_aligned_noclip.wav')
goto_dir = os.path.join('D:/data/speech_Southmead/audio/','wavs/')

file_list = []
total_seconds = 0
seg = 512
sf = 48000 # sampling frequency of wav file # (recorded at 16 bit, 48 KHz)

audio, sf = load(wav, sr=sf, mono=False)
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,1)

track = 0
seen = []
for i in mixdown_order:
    seen.append(i)
    file_path = goto_dir+str(i)+'-'+str(seen.count(i))+'.wav' #os.path.join(goto_dir, f'{i}-{seen.count(i)}.wav')
    part = audio[track:track+(sf*5)]
    track+=(sf*5)
    soundfile.write(file_path, part, sf)


selected = ['1-3.wav', '2-2.wav', '3-3.wav', '4-3.wav', '5-2.wav', '6-3.wav', '7-2.wav', '8-3.wav', '9-2.wav', '10-2.wav', '11-4.wav', '12-4.wav', '13-2.wav', '14-2.wav', '15-3.wav', '16-3.wav', '17-2.wav', '18-2.wav', '19-1.wav', '20-1.wav', '21-2.wav', '22-2.wav', '23-1.wav', '24-2.wav', '25-1.wav', '26-2.wav', '27-2.wav', '28-3.wav', '29-2.wav', '30-3.wav', '31-2.wav', '32-2.wav', '33-1.wav', '34-2.wav', '35-1.wav', '36-1.wav', '37-1.wav', '38-2.wav', '39-2.wav', '40-2.wav', '41-1.wav', '42-3.wav', '43-2.wav', '44-3.wav', '45-2.wav', '46-1.wav', '47-2.wav', '48-1.wav', '49-3.wav', '50-3.wav', '51-1.wav', '52-3.wav', '53-2.wav', '54-3.wav', '55-1.wav', '56-1.wav', '57-2.wav', '58-2.wav', '59-3.wav', '60-1.wav', '61-1.wav', '62-2.wav', '63-1.wav', '64-1.wav', '65-1.wav', '66-1.wav', '67-3.wav', '68-1.wav', '69-1.wav', '70-3.wav', '71-3.wav', '72-1.wav', '73-2.wav', '74-1.wav', '75-3.wav', '76-2.wav', '77-3.wav', '78-1.wav', '79-2.wav', '80-3.wav', '81-3.wav', '82-1.wav', '83-2.wav', '84-2.wav', '85-2.wav', '86-2.wav', '87-3.wav', '88-3.wav', '89-1.wav', '90-3.wav', '91-3.wav', '92-1.wav', '93-1.wav', '94-3.wav', '95-1.wav', '96-3.wav', '97-1.wav', '98-1.wav', '99-2.wav', '100-3.wav']
file_list = []
file_list.append(np.zeros(int(sf)*5))
for i in selected:
    wav = goto_dir+str(i)#os.path.join(goto_dir, i)
    audio, sf = load(wav, sr=sf, mono=False) # mono=True converts stereo audio to mono
    file_list.append(audio)
    file_list.append(np.zeros(int(sf)*10))
audio = np.concatenate(file_list)
stim_length = (sf*5*3*100)/sf
prestim_length = (sf*5)/sf
print(f'Output total samples: {audio.shape[0]}')
print(f'Output is expected length ({stim_length}s stim + {prestim_length}s prestim = {stim_length+prestim_length}s): {audio.shape[0]==(sf*5*3*100)+(sf*5)}')
print(f'Output is of length: {audio.shape[0]/sf} seconds.')
#print(f'(5 seconds pre-stimulus alignment audio + {int((audio.shape[0]/sf)-5)} seconds stimulus audio).')
soundfile.write(goto_dir+'output.wav', audio, sf)

file_list = []
total_seconds = 0
seg = 512
sf = 48000  # sampling frequency of wav file # (recorded at 16 bit, 48 KHz)
# half_second_bit = int((((sf/2)/10)/2))
# quarter_second_bit = int((((sf/4)/10)/2))
# eighth_second_bit = int((((sf/8)/10)/2))
# tenth_second_bit = int((((sf/10)/10)/2))

def overlay_audio(audio1_path, audio2_path):
    from pydub import AudioSegment

    sound1 = AudioSegment.from_file(audio1_path)
    sound2 = AudioSegment.from_file(audio2_path)

    combined = sound1.overlay(sound2)

    combined.export("combined.wav", format='wav')


def insert_pulse(audio, binary, bit_length=2):
    audio = copy.deepcopy(audio)
    pos = 4.999999999999999
    neg = -4.999999999999999
    binary = binary[2:]
    bit = int((((sf / bit_length) / 10) / 2))
    t = 0
    for i in binary:
        if bool(int(i)):
            audio[t:t + bit] = pos
        #         else:
        #             audio[t:t+bit] = neg
        t += (bit * 2)
    return audio


def insert_pulse_reverse(audio, binary, bit_length=10):
    audio = copy.deepcopy(audio)
    pos = 4.999999999999999
    neg = -4.999999999999999
    audio = audio[::-1]
    binary = binary[2:]
    bit = int((((sf / bit_length) / 10) / 2))
    t = 0
    for i in binary:
        if bool(int(i)):
            audio[t:t + bit] = pos
        #         else:
        #             audio[t:t+bit] = neg
        t += (bit * 2)
    audio = audio[::-1]
    return audio


tone0 = np.full(int(sf), 0)  # a second of silence

tone1 = librosa.clicks(times=np.array([.4]), sr=sf)  # a half-second click
tone8 = librosa.tone(librosa.note_to_hz('C6'), sr=sf, duration=0.1)  # used just to get a 100ms audio length
tone2 = librosa.tone(librosa.note_to_hz('C5'), sr=sf, duration=0.2)  # used just to get a 100ms audio length
tone3 = librosa.tone(librosa.note_to_hz('C4'), sr=sf, duration=0.5)  # the actual tones used for alignment
tone4 = librosa.tone(librosa.note_to_hz('C6'), sr=sf, duration=0.5)  # used just to get a 100ms audio length
tone5 = librosa.tone(librosa.note_to_hz('C3'), sr=sf, duration=0.95)  # used just to get a 1000ms audio length
tone6 = librosa.tone(librosa.note_to_hz('C2'), sr=sf, duration=1.0)  # used just to get a 1000ms audio length
tone7 = librosa.tone(librosa.note_to_hz('C1'), sr=sf, duration=0.9)  # used just to get a 1000ms audio length

metronome = librosa.clicks(times=[0], frames=None, sr=48000, click_freq=2056.0,
                           click_duration=0.05) / 2  # , click=None, length=None)
metronome = np.hstack((metronome, np.zeros(tone5.shape[0])))

# t=0
file_list.append(np.zeros(int(sf)))
# t=1000
file_list.append(np.zeros(int(sf)))
# t=2000
file_list.append(tone2 / 4)
file_list.append(np.zeros(int(sf * .8)))
# t=3000
file_list.append(tone2 / 4)
file_list.append(np.zeros(int(sf * .8)))
# t=4000
file_list.append(tone2 / 4)
file_list.append(np.zeros(int(sf * .8)))
# t=5000

for i in range(100):
    file_list.append(librosa.tone(librosa.note_to_hz('C6'), sr=sf, duration=0.2) / 4)
    #file_list.append(np.zeros(int(sf * .8))) # replace begining with square wave
    file_list.append(np.ones(int(sf * .1)))
    file_list.append(np.zeros(int(sf * .7)))

    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(librosa.tone(librosa.note_to_hz('C5'), sr=sf, duration=0.2) / 4)
    file_list.append(np.zeros(int(sf * .8)))
    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(librosa.tone(librosa.note_to_hz('C5'), sr=sf, duration=0.2) / 4)
    file_list.append(np.zeros(int(sf * .8)))
    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(metronome)
    file_list.append(metronome)
    seg += 1

import sounddevice as sd
fs=48000
sd.play(audio[:60*sf],fs)
sd.stop()

audio = np.concatenate(file_list)
stim_length = (sf * 5 * 3 * 100) / sf
prestim_length = (sf * 5) / sf
print(f'Output total samples: {audio.shape[0]}')
print(
    f'Output is expected length ({stim_length}s stim + {prestim_length}s prestim = {stim_length + prestim_length}s): {audio.shape[0] == (sf * 5 * 3 * 100) + (sf * 5)}')
print(f'Output is of length: {audio.shape[0] / sf} seconds.')
# print(f'(5 seconds pre-stimulus alignment audio + {int((audio.shape[0]/sf)-5)} seconds stimulus audio).')
soundfile.write(goto_dir+'output2.wav', audio, sf)


def overlay_audio(audio1_path, audio2_path, output=''):
    from pydub import AudioSegment
    sound1 = AudioSegment.from_file(audio1_path)
    sound2 = AudioSegment.from_file(audio2_path)
    combined = sound1.overlay(sound2)
    combined.export(output, format='wav')

output = goto_dir+'output.wav'#os.path.join(os.getcwd(), 'output.wav')
output2 = goto_dir+'output2.wav'# os.path.join(os.getcwd(), 'output2.wav')
overlay_audio(output, output2, output=goto_dir+'output3.wav')


sf = 48000
audio, sf = load(goto_dir+'output3.wav', sr=sf, mono=False)  # mono=True converts stereo audio to mono
total_seconds = 0
seg = 512

sf = 48000  # sampling frequency of wav file # (recorded at 16 bit, 48 KHz)
t = sf * 5  # 5 seconds in
pos = 4.999999999999999
neg = -4.999999999999999
bit = int(((sf / 5) / 20))

for i in range(300):
    binary = str(bin(seg))[2:]
    for j in binary:
        if bool(int(j)):
            audio[t:t + bit] = pos
        t += (bit * 2)
    t = (((i + 1) * sf) * 5) + (sf * 5)
    seg += 1

soundfile.write('output5.wav', audio, sf)



aa=r'E:\Bristol\experiment\imagery_speech_English_Southmead\audio\square_wave\5_second_wavs\1.wav'
audio, sf = load(aa, sr=sf)





