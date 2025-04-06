
url = 'https://www.youtube.com/watch?v=snkwsU98QlQ'
url = 'https://upload.wikimedia.org/wikipedia/commons/d/de/Hello_in_Morse_Code.wav'
url = 'https://commons.wikimedia.org/wiki/File:1210secretmorzecode.wav'

import requests
requests.get?
wav = requests.get(url)
wav.content
url = 'https://upload.wikimedia.org/wikipedia/commons/7/78/1210secretmorzecode.wav'
wav = requests.get(url)
wav
wav.content
headers = {'User-Agent': 'NLPiA2Bot/0.7 (https://tangibleai.com/about/; nlpia2@tangibleai.com)'}
wav = requests.get(url, headers=headers)
wav
wav.content
bytearray(wav.content)
resp = wav
wav = pd.Series([b for b in resp.content])
import pandas as pd
wav = pd.Series([b for b in resp.content])
wav
wav = pd.Series(resp.content)
wav
chr(82
    )
from urllib.request import urlretrieve
f = urlretrieve(url, '1210secretmorzecode.wav')
ls - hal
from scipy.io import wavfile
samplerate, data = wavfile.read('1210secretmorzecode.wav')
data
wav = pd.Series(data)
wavfile?
dir(wavfile)
dir(wavfile)
wav = pd.Series(data)
wav.plot()
from seaborn import pyplot as plt
from matplotlib import pyplot
from matplotlib import pyplot as plt
plt.show()
wav.describe()
pwd
wav
from scipy.io import wavfile
samplerate, data = wavfile.read('1210secretmorzecode.wav')
samplerate
data
data[0]
data.shape
import wave
import struct

wavefile = wave.open('sine.wav', 'r')

length = wavefile.getnframes()
for i in range(0, length):
    wavedata = wavefile.readframes(1)
    data = struct.unpack("<h", wavedata)
    print(int(data[0]))
filename = '1210secretmorzecode.wav'
import wave
import struct

wavefile = wave.open(filename, 'r')

length = wavefile.getnframes()
for i in range(0, length):
    wavedata = wavefile.readframes(1)
    data = struct.unpack("<h", wavedata)
    print(int(data[0]))
wavedata
import wave
import struct

wavefile = wave.open(filename, 'r')

length = wavefile.getnframes()
for i in range(0, length):
    wavedata = wavefile.readframes(2)
    data = struct.unpack("<h", wavedata)
    print(int(data[0]))
!pip install SoundFile
sf = SoundFile(filename)
import SoundFile as sf
import soundfile as sf
sf.read(filename)
wav = sf.read(filename)
wav.shape
wav
wav[0]
wav[0].shape
wav[0]
wav[0][1:2:]
wav[0][1::2]
wav[0][0::2]
pd.Series(wav[0][0::2]).plot()
plt.show()
pd.Series(wav[0][0::2][:10000]).plot()
plt.show()
pd.Series(wav[0][:10000]).plot()
plt.show()
wavedata
wav = pd.Series(resp.content)
wav
from scipy.io import wavfile
samplerate, data = wavfile.read('1210secretmorzecode.wav')
wav = pd.Series(data)
wav = pd.Series(data[:5000])
wav.plot()
plt.show()
wav = pd.Series(data[:10000])
wav.plot()
plt.show
plt.show()
samplerate
wav.index = wav.index / samplerate
wav.plot()
plt.show()
plt.show()
wav
wav.plot()
plt.show()
wav.plot()
plt.show()
pwd
hist - o - p - f code / tangibleai / nlpia2 / src / nlpia2 / ch07 / download_read_plot_wav_file.hist.md
hist - f code / tangibleai / nlpia2 / src / nlpia2 / ch07 / download_read_plot_wav_file.hist.py
