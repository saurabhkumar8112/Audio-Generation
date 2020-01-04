from scipy import misc
import imageio
import numpy as np 
import pandas as pd 
import os
from natsort import natsorted
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from skimage.measure import block_reduce


def to_rgb(dir):
	arr=[]
	os.chdir(dir)
	for file in natsorted(os.listdir(dir)):
		img=imageio.imread(file)
		img=block_reduce(img,block_size=(3,3,1),func=np.max)
		arr.append(img)
	return np.array(arr)

def to_rgb_fun(figures):
	'''figures=[]
	os.chdir(dir)
	for file in os.listdir(dir):
		img=imageio.imread(file)
		print(img.shape)
		figures.append(img)'''
	fig,axes=plt.subplots(figsize=(7,7),nrows=2, ncols=2, sharey=True, sharex=True)
	for ax,img in zip(axes.flatten(),figures):
		ax.xaxis.set_visible(True)
		ax.yaxis.set_visible(True)
		image=Image.fromarray(img,'RGB')
		img=ax.imshow(image)
	plt.show()
	'''img=imageio.imread(file)
	print(img)
	image = Image.fromarray(img,'RGB' )
	plt.imshow(image)
	plt.show()'''

#to_rgb_fun("C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Resources/For_Plotting")

'''images=to_rgb("C:/Users/Saurabh Kumar/Desktop/ALL PROJECTS/Audio Generation-PAP/Spectrograms")
print(images.shape)'''