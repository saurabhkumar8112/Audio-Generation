import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.pyplot import specgram
import wavio
from scipy import fft
import os
from audio_slicing import checker
from natsort import natsorted

"""
Globale Variable definition
"""
NFFT=1024
noverlap=100
def gen_spectrogram(dir,saving_dir):
    counter=len(os.listdir(dir))
    min_file_name,min_file_size,max_file_name,max_file_size=checker(dir)
    for file in natsorted(os.listdir(dir)):
        print("converting to spectrogram, file name "+file)
        spec_name=str(file)
        spec_name=spec_name[:-4]
        print("saving to "+spec_name+".png")
        os.chdir(dir)
        sound=wavio.read(file)
        data,rate,samplewidth=sound.data,sound.rate,sound.sampwidth
        print("datashape: {} sampling_rate:{} samplewidth: {}".format(len(data),rate,samplewidth))
        y=fft(data)/len(data)
        y=np.real(y[:,0])
        y=[y[i] for i in range(len(y)) if y[i]!=0]
        y=y[:min_file_size]
        print(len(y))
        pxx,freqs,bins,im=specgram(y,NFFT=NFFT,Fs=rate,noverlap=100)
        os.chdir(saving_dir)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        print("Saving File\n")
        plt.savefig(spec_name+".jpg")
        print("Remaining files {}".format(counter))
        #plt.show()
        counter-=1
'''def gen_spectrogram_from_file(file,dir,saving_dir):
    min_file_name,min_file_size,max_file_name,max_file_size=checker(dir)
    print("converting to spectrogram, file name "+file)
    spec_name=str(file)
    spec_name=spec_name[:-4]
    print("saving to "+spec_name+".png")
    os.chdir(dir)
    sound=wavio.read(file)
    data,rate,samplewidth=sound.data,sound.rate,sound.sampwidth
    print("datashape: {} sampling_rate:{} samplewidth: {}".format(len(data),rate,samplewidth))
    y=fft(data)/len(data)
    #y=y[:min_file_size]
    y=np.real(y[:,0])
    print(len(y))
    y=[y[i] for i in range(len(y)) if y[i]!=0]
    pxx,freqs,bins,im=specgram(y,NFFT=NFFT,Fs=rate,noverlap=100)
    os.chdir(saving_dir)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    print("Saving File\n")
    plt.savefig(spec_name+".png")
    plt.show()'''



gen_spectrogram("C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Violin_Music/WAV","C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Spectrograms")
#gen_spectrogram_from_file("142.wav","C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Violin_Music/WAV",
 #   "C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Spectrograms")
