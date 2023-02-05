import time
import tracemalloc
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft

# utility functions
def load_wav(file):
    samplerate, data = wavfile.read(file)
    print("{0}: {1}Hz".format(file, samplerate))
    if data.ndim > 1:
        data = data[:,1]
    data = data.astype(np.float32)
    data = data/max(abs(data))
    return data

def pad_zeros_to(input, new_length):
    output = np.zeros((new_length,), dtype=np.float32)
    output[:input.shape[0]] = input
    return output

def next_power_of_2(n):
    return 1 << (int(np.log2(n - 1)) + 1)

# overlap add
def process_block_ola(x, Y, K, N, buffer):
    B = len(x)
    
    # perform fft of input block
    X = fft(x, K)

    # spectral convolution
    Z = np.multiply(X, Y)

    # inverse fft
    z = np.real(ifft(Z))

    # add to buffer
    # have to pad zeros so same size
    z = pad_zeros_to(z[:B+N-1], len(buffer))
    buffer += z

    # take output from buffer
    output = buffer[:B]

    # shift buffer left
    for i in range(0, len(buffer)-B, B):
        buffer[i:i+B] = buffer[i+B:i+(2*B)]

    # zero end of buffer
    buffer[-B:] = 0

    return output, buffer

def overlap_add_time(x, y, B):
    # allocating memory buffers and calculating variables

    # handle the case where x doesn't divide equally into blocks (won't happen running)
    num_blocks = np.ceil(len(x)/B).astype(int)
    x = pad_zeros_to(x, num_blocks*B)

    blocks = [x[i:i+B] for i in range(0, len(x), B)]

    # work out K
    K = next_power_of_2(B + len(y) - 1)

    # assume filter will have been already fft
    Y = fft(y, K)

    num_buffer_blocks = np.ceil((B+len(y)-1)/B).astype(int)
    buffer = np.zeros((num_buffer_blocks*B,), dtype=np.float32)

    times = []

    for block in blocks:
        t_start = time.time()
        block_output, buffer = process_block_ola(block, Y, K, len(y), buffer)
        t_end = time.time()
        times.append(t_end-t_start)

    return times

# overlap save
def process_block_ols(x, Y, K, N, buffer):
    B = len(x)

    # update buffer
    for i in range(0, len(buffer)-B, B):
        buffer[i:i+B] = buffer[i+B:i+(2*B)]
    buffer[-B:] = x

    # perform fft of buffer
    X = fft(buffer, K)
    
    # spectral convolution
    Z = np.multiply(X, Y)

    # inverse fft
    z = np.real(ifft(Z))
    return z[-B:], buffer

def overlap_save_time(x, y, B):
    # allocating memory buffers and calculating variables

    # handle the case where x doesn't divide equally into blocks (won't happen running)
    num_blocks = np.ceil(len(x)/B).astype(int)
    x = pad_zeros_to(x, num_blocks*B)

    blocks = [x[i:i+B] for i in range(0, len(x), B)]

    # work out K
    K = next_power_of_2(B + len(y) - 1)

    # assume fft is already available so don't time
    Y = fft(y, K)

    # K should be divisible by B as they should both be powers of 2 but deal with edge case
    num_buffer_blocks = np.ceil(K/B).astype(int)
    buffer = np.zeros((num_buffer_blocks*B,), dtype=np.float32)

    times = []
    
    for block in blocks:
        t_start = time.time()
        block_output, buffer = process_block_ols(block, Y, K, len(y), buffer)
        t_end = time.time()
        times.append(t_end-t_start)
    
    return times

# uniform partition
def process_block_up_ols(x, Y_blocks, buffer, fdl):
    B = len(x)
    
    buffer[:B] = buffer[-B:]
    buffer[-B:] = x

    # fft of buffer
    X = fft(buffer)
    # X = X[:B+1]
    # update fdl
    for i in range(fdl.shape[0]-1):
        fdl[i] = fdl[i+1]
    fdl[fdl.shape[0]-1] = X

    output = np.zeros((2*B,), dtype=np.complex64)

    for i in range(fdl.shape[0]):
        Z = np.multiply(fdl[fdl.shape[0]-1-i], Y_blocks[i])
        output += Z
    
    out = np.real(ifft(output, 2*B))

    return out[-B:], buffer

def up_ols_time(x, y, B):
    # allocating memory buffers and calculating variables
    # handle the case where x or y doesn't divide equally into blocks (will be predone)
    num_x_blocks = np.ceil(len(x)/B).astype(int)
    x = pad_zeros_to(x, num_x_blocks*B)
    num_y_blocks = np.ceil(len(y)/B).astype(int)
    y = pad_zeros_to(y, num_y_blocks*B)

    x_blocks = [x[i:i+B] for i in range(0, len(x), B)]
    y_blocks = [y[i:i+B] for i in range(0, len(y), B)]

    # presume fft is already done
    Y_blocks = [fft(block, 2*B) for block in y_blocks]

    buffer = np.zeros((2*B,), dtype=np.float32)
    
    # size of fdl can be reduced to B+1
    fdl = np.zeros((num_y_blocks, 2*B), dtype=np.complex64)

    times = []

    for block in x_blocks:
        t_start = time.time()
        block_output, buffer = process_block_up_ols(block, Y_blocks, buffer, fdl)
        t_end = time.time()
        times.append(t_end-t_start)
    
    return times


if __name__ == "__main__":
    # load in required data
    data = load_wav("gs/Alesis-Fusion-Clean-Guitar-C3.wav")
    reverb = load_wav("irs/1st_baptist_nashville_balcony.wav")
    delay_ir = load_wav("irs/GT-8 Impulse Responses 1.01/GT-8 Tape Delay L.wav")

    # test with block sizes from 64 - 4096
    block_sizes = [2**i for i in range(6, 13)]

    # convolution tests
    # reverb ir size
    

    for i in range(16000, 288001, 16000):
        print("reverb size: {0}".format(i))
        reverb_ir = reverb[:i]
        # test overlap_add
        print("--------- ola ---------")
        for s in block_sizes:
            avg = []
            total = []
            for i in range(50):
                t = overlap_add_time(data, reverb_ir, s)
                avg.append(np.mean(t)*1000)
                total.append(np.sum(t))
            print("B = {0}: {1}s total ({2}ms / block)".format(s, np.mean(total), np.mean(avg)))
        print("")
        
        # test overlap_save
        print("--------- ols ---------")
        for s in block_sizes:
            avg = []
            total = []
            for i in range(50):
                t = overlap_save_time(data, reverb_ir, s)
                avg.append(np.mean(t)*1000)
                total.append(np.sum(t))
            print("B = {0}: {1}s total ({2}ms / block)".format(s, np.mean(total), np.mean(avg)))
        print("")
        
        # test uniform partition
        print("-------- upols --------")
        for s in block_sizes:
            avg = []
            total = []
            for i in range(50):
                t = up_ols_time(data, reverb_ir, s)
                avg.append(np.mean(t)*1000)
                total.append(np.sum(t))
            print("B = {0}: {1}s total ({2}ms / block)".format(s, np.mean(total), np.mean(avg)))
        print("")
    

