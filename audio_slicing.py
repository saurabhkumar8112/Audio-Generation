import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.pyplot import specgram
import wavio
from scipy import fft
import os
from natsort import natsorted


def checker(dir):
	os.chdir(dir)
	file_dict={}
	file_size_arr=[]
	for file in natsorted(os.listdir(dir)):
		sound=wavio.read(file)
		data,rate,samplewidth=sound.data,sound.rate,sound.sampwidth
		file_dict[len(data)]=str(file)
		file_size_arr.append(len(data))
	return file_dict[np.min(file_size_arr)],np.min(file_size_arr),file_dict[np.max(file_size_arr)],np.max(file_size_arr)

'''file_dict,file_size_arr=checker("C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Violin_Music/WAV")
file_size_arr=np.array(file_size_arr)
print("File name, {}, length {} \nFile name, {},length {}".format
	())'''


