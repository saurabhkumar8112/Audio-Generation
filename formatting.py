import os

def format(dir):
	c=1
	for file in os.listdir(dir):
		os.chdir(dir)
		os.rename(file,str(c)+".wav")
		c+=1
format("C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Violin_Music/WAV")