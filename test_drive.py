import time
import numpy as np
from scipy.io import wavfile
import math

def load_wav(file):
    samplerate, data = wavfile.read(file)
    print("{0}: {1}Hz".format(file, samplerate))
    if data.ndim > 1:
        data = data[:,1]
    data = data.astype(np.float32)
    data = data/max(abs(data))
    return data

# pg. 124-125
def overdrive(data, drive):
    data = data * drive
    output = np.zeros((len(data),), dtype=np.float32)
    for i in range(len(data)):
        if abs(data[i]) < 1/3:
            output[i] = 2*data[i]
        elif abs(data[i]) < 2/3:
            output[i] = (3 - (2-3*abs(data[i]))**2 )/(3)
            output[i] = np.copysign(output[i], data[i])
        else:
            output[i] = np.copysign(1, data[i])

    for i in range(len(data)):
            if abs(output[i]) > 1:
                print("i: {0}, data: {1}".format(i, data[i]))
    return output

# https://dsp.stackexchange.com/questions/13142/digital-distortion-effect-algorithm
def distortion(data, drive):
    data = data * drive
    output = np.zeros((len(data),), dtype=np.float32)
    for i in range(len(data)):
        output[i] = np.copysign(1-np.exp(-np.abs(data[i])), data[i])
    return output

def clipper(data, drive):
    data = data * drive
    output = np.zeros((len(data),), dtype=np.float32)
    for i in range(len(data)):
        output[i] = data[i]/(1+abs(data[i]))
    return output

# https://z2dsp.com/2017/09/04/modelling-fuzz/
# need to implement a envelope follower for this to work properly
def fuzz(data, drive):
    data = data + 0.8
    data = data * drive
    output = np.zeros((len(data),), dtype=np.float32)
    for i in range(len(data)):
        output[i] = data[i]/(1+abs(data[i]))
    return output


# also need to implement some filtering in this effect
if __name__ == "__main__":
    data = load_wav("gs/audio.wav").astype(np.float32)
    output = clipper(data, 3)

    wavfile.write("test_od.wav", 44100, output)