# Audio-Generation
Using GANs to generate Audios. GANs have shown tremendous capabilities to learn data distribution and generate data from the same. In this project I explore the possibilities of a GAN to generate simple harmonic musical sound of an instrument, for eg. violin in this case.
So basically I am training a GAN to produce violin sounds. Pretty cool huhh. Now the big problem here is, how shall I feed the sound to a GAN? I cannot simply feed a MP3 file. I have to take a mathemtical decomposition of the same. 

One thing that could have been done is to take only the amplitude value of the sound but this doesn't take into account the phase of a musical sound. So here we use something different. A spectrogram, which is a graphical representation of an audio signal. Like show here.

![alt text](https://raw.githubusercontent.com/saurabhkumar8112/Audio-Generation/edit/master/Spectrograms/1.JPG)


