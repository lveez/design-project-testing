import numpy as np
from scipy.io import wavfile

def load_wav(file):
    samplerate, data = wavfile.read(file)
    print("{0}: {1}Hz".format(file, samplerate))
    if data.ndim > 1:
        data = data[:,1]
    data = data.astype(np.float32)
    data = data/max(abs(data))
    return data

# https://picture.iczhiku.com/resource/eetop/sykfGloQfYJRZBcv.pdf pp. 62-65
def lowshelving(data, xh, fc, fs, gain):
    V0 = 10**(gain/20)
    H0 = V0 - 1
    if gain < 0:
        c = (np.tan(np.pi*fc/fs) - V0)/(np.tan(np.pi*fc/fs) + V0)
    else:
        c = (np.tan(np.pi*fc/fs) - 1)/(np.tan(np.pi*fc/fs) + 1)
    
    
    y = np.zeros((len(data),), dtype=np.float32)
    out = np.zeros((len(data),), dtype=np.float32)

    for i in range(1, len(data)):
        xh[i] = data[i] - (c * xh[(i-1) % len(data)])
        y[i] = (c * xh[i]) + xh[(i-1) % len(data)]
        out[i] = (H0/2 * (data[i] + y[i]))
    
    return out, xh

def highshelving(data, xh, fc, fs, gain):
    V0 = 10**(gain/20)
    H0 = V0 - 1
    if gain < 0:
        c = (np.tan(np.pi*fc/fs) - V0)/(np.tan(np.pi*fc/fs) + V0)
    else:
        c = (np.tan(np.pi*fc/fs) - 1)/(np.tan(np.pi*fc/fs) + 1)
    
    y = np.zeros((len(data),), dtype=np.float32)
    out = np.zeros((len(data),), dtype=np.float32)

    for i in range(len(data)):
        xh[i] = data[i] - (c * xh[(i-1) % len(data)])
        y[i] = (c * xh[i]) + xh[(i-1) % len(data)]
        out[i] = ((H0/2) * (data[i] - y[i]))
    
    return out, xh

def peakfilter(data, xh, fc, fband, fs, gain):
    V0 = 10**(gain/20)
    H0 = V0 - 1
    d = np.cos(2*np.pi*fc/fs)

    if gain < 0:
        c = (np.tan(np.pi*fband/fs) - V0)/(np.tan(np.pi*fband/fs) + V0)
    else:
        c = (np.tan(np.pi*fband/fs) - 1)/(np.tan(np.pi*fband/fs) + 1)

    y = np.zeros((len(data),), dtype=np.float32)
    out = np.zeros((len(data),), dtype=np.float32)

    for i in range(len(data)):
        xh[i] = data[i] - (d*(1-c)*xh[(i-1) % len(data)]) + (c*xh[(i-2) % len(data)])
        y[i] = (-c*xh[i]) + (d*(1-c)*xh[(i-1) % len(data)]) + xh[(i-2) % len(data)]
        out[i] = ((H0/2) * (data[i]-y[i]))

    return out, xh


if __name__ == "__main__":
    data = load_wav("gs/audio.wav").astype(np.float32)
    xh = np.zeros((len(data,)), dtype=np.float32)
    outputlow, _ = lowshelving(data, xh, 300, 44100, 0)
    outputmid, _ = peakfilter(data, xh, 1000, 1400, 44100, 0)
    outputhigh, _ = highshelving(data, xh, 1700, 44100, 0)
    
    output = data + outputlow + outputmid + outputhigh
    print(max(abs(output)))

    wavfile.write("test_eq2.wav", 44100, output)