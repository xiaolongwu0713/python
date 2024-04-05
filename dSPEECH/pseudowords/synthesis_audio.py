'''
This script is to synthesize speech audio from text (TTS) using Coqui_AI;
Use the synthesis_audio.sh instead.

'''

## method 1
from TTS.api import TTS
tts=TTS(model_name='tts_models/ga/cv/vits')
tts.tts_to_file(text='neep kootorm ang.')


## method 2
import requests
import sounddevice as sd

API_URL = "https://api-inference.huggingface.co/models/voices/VCTK_British_English_Males"
headers = {"Authorization": f"Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

audio_bytes = query({"inputs": "The answer to the universe is 42",})
# You can access the audio with IPython.display for example
from IPython.display import Audio
Audio(audio_bytes)

fs=16000
sd.play(audio_bytes,fs)
sd.stop()


