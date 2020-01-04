"""Generate a Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""
import os
import numpy as np 
import wave
from pydub import AudioSegment
import ffmpeg
import os
print(os.listdir())
print(os.getcwd())
import pylab
from scipy.io import wavfile
import matplotlib.pyplot as plot
 

# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read('Violin_Music/WAV/violin.wav')

 

# Plot the signal read from wav file

plot.subplot(211)

plot.title('Spectrogram of a wav file with piano music')

 

plot.plot(signalData)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.show()

plot.subplot(212)
print(signalData.shape)
plot.specgram(signalData[:,0],Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')

 

plot.show()
'''def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    print((sound_info,frame_rate))
    NFFT=200
    pylab.figure(num=None, figsize=(10, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate,NFFT=NFFT,noverlap=0.0001)
    pylab.show()
    pylab.savefig('spectrogram.png')
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'rb')
    frames = wav.readframes(-1)
    print(frames,wav.getframerate())
    print(type(wav))
    sound_info = pylab.fromstring(frames, 'Int8')
    sound_info=sound_info.astype(np.uint16)
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
if __name__ == '__main__':
    wav_file = 'Violin_Music/WAV/356181__mtg__violin-d5.wav'
    graph_spectrogram(wav_file)'''

"""Generate a Spectrogram image for a given audio sample.

Compatible with several audio formats: wav, flac, mp3, etc.
Requires: https://code.google.com/p/timeside/

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""
'''import timeside
audio_file = 'Sample.mp3'

decoder = timeside.decoder.FileDecoder(audio_file)
grapher = timeside.grapher.Spectrogram(width=1920, height=1080)
(decoder | grapher).run()
grapher.render('spectrogram.png')'''


'''class Spectrogram:
	def con_mp3(self,directory):

	def gen_spec(self,directory,frame_rate=44100):'''

#file=open('Violin_Music/Sample.mp3')

def to_mp3(dir):
	for file in os.listdir(dir):
		print(file)
		file_read="Violin_Music/MP3/"+file
		s="Violin_Music/WAV/"
		s=s+str(file)
		print(s)
		s=s[:-3]+"wav"
		sound=AudioSegment.from_mp3(file_read)
		sound.export(s,format="wav")
'''sound=AudioSegment.from_mp3('Violin_Music/Sample.mp3')
sound.export('Violin_Music/Sample1.wav',format="wav")'''