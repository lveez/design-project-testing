import tracemalloc
import time
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft

def pad_zeros_to(input, new_length):
    output = np.zeros((new_length,))
    output[:input.shape[0]] = input
    return output

def next_power_of_2(n):
    return 1 << (int(np.log2(n - 1)) + 1)

def load_wav(file):
    samplerate, data = wavfile.read(file)
    if data.ndim > 1:
        data = data[:,1]
    data = data.astype(np.float32)
    data = data/max(abs(data))
    return data

# this uses overlap add only partioning the signal
def overlap_add(x, y, B, test=False):
    '''
    x - input signal
    y - filter
    B - block length
    '''

    # handle the case where x doesn't divide equally into blocks (won't happen running)
    num_blocks = np.ceil(len(x)/B).astype(int)
    x = pad_zeros_to(x, num_blocks*B)

    blocks = [x[i:i+B] for i in range(0, len(x), B)]

    # work out K
    K = next_power_of_2(B + len(y) - 1)

    tracemalloc.start()

    # perform fft of filter
    Y = fft(y, K)

    s, p = tracemalloc.get_traced_memory()
    print(p/1024)

    num_buffer_blocks = np.ceil((B+len(y)-1)/B).astype(int)
    buffer = np.zeros((num_buffer_blocks*B,), dtype=np.float32)
    
    # storing output so can listen back to it, remove when testign memory performance
    if test == False:
        output = []
        original = []
        # process each block individually
        for block in blocks:
            block_output, buffer = process_block_ola(block, Y, K, len(y), buffer)
            # also remove these when testing performance
            output = np.concatenate((output, block_output)).astype(np.float32)
            original = np.concatenate((original, block)).astype(np.float32)
    else:
        t_start = time.time()
        for block in blocks:
            # _, p = tracemalloc.get_traced_memory()
            # print(p/1024)
            block_output, buffer = process_block_ola(block, Y, K, len(y), buffer)
        t_end = time.time()

    if test:
        # return time in seconds of processing / second of input signal
        return (t_end-t_start)/len(x), p/1024
    
    return output, original


def process_block_ola(x, Y, K, N, buffer):
    B = len(x)
    
    # perform fft of input block
    X = np.fft.fft(x, K)

    # spectral convolution
    Z = np.multiply(X, Y)

    # inverse fft
    z = np.real(np.fft.ifft(Z))
    z = z[:B+N-1] # only care about first B + N - 1 samples

    # add to buffer
    # have to pad zeros so same size
    z = pad_zeros_to(z, len(buffer))
    buffer += z

    # take output from buffer
    output = buffer[:B]

    # shift buffer left
    for i in range(0, len(buffer)-B, B):
        buffer[i:i+B] = buffer[i+B:i+(2*B)]

    # zero end of buffer
    buffer[-B:] = 0

    return output, buffer

# this uses overlap save only partitioning the signal
def overlap_save(x, y, B):
    '''
    x - input signal
    y - filter
    B - block length
    '''

    # handle the case where x doesn't divide equally into blocks (won't happen running)
    num_blocks = np.ceil(len(x)/B).astype(int)
    x = pad_zeros_to(x, num_blocks*B)

    blocks = [x[i:i+B] for i in range(0, len(x), B)]

    # work out K
    K = next_power_of_2(B + len(y) - 1)

    # perform fft of filter
    Y = np.fft.fft(y, K)    # np.fft auto zero pads

    # K should be divisible by B as they should both be powers of 2 but deal with edge case
    num_buffer_blocks = np.ceil(K/B).astype(int)
    buffer = np.zeros(num_buffer_blocks*B)

    # storing output so can listen back to it, remove when testign memory performance
    output = []
    original = []

    # process each block individually
    for block in blocks:
        block_output, buffer = process_block_ols(block, Y, K, len(y), buffer)
        output = np.concatenate((output, block_output)).astype(np.float32)
        original = np.concatenate((original, block)).astype(np.float32)

    return output, original

def process_block_ols(x, Y, K, N, buffer):
    B = len(x)

    # update buffer
    for i in range(0, len(buffer)-B, B):
        buffer[i:i+B] = buffer[i+B:i+(2*B)]
    buffer[-B:] = x

    # perform fft of buffer
    X = np.fft.fft(buffer, K)
    
    # spectral convolution
    Z = np.multiply(X, Y)

    # inverse fft
    z = np.real(np.fft.ifft(Z))
    return z[-B:], buffer



# this uses overlap save with uniform partitioning i.e. both
# filter and signal is partitioned
def uniform_partition_ols(x, y, B):
    '''
    x - input signal
    y - filter
    B - block length
    '''
    # handle the case where x or y doesn't divide equally into blocks (won't happen running)
    num_x_blocks = np.ceil(len(x)/B).astype(int)
    x = pad_zeros_to(x, num_x_blocks*B)
    num_y_blocks = np.ceil(len(y)/B).astype(int)
    y = pad_zeros_to(y, num_y_blocks*B)

    x_blocks = [x[i:i+B] for i in range(0, len(x), B)]
    y_blocks = [y[i:i+B] for i in range(0, len(y), B)]
    Y_blocks = [np.fft.fft(block, 2*B) for block in y_blocks] # numpy automatically zero pads fft
    
    # # removing unneeded elements to speed up multiplication
    # for i in range(len(Y_blocks)):
    #     Y_blocks[i] = Y_blocks[i][:B+1]

    buffer = np.zeros(2*B)
    # size of fdl can be reduced to B+1
    fdl = np.zeros((num_y_blocks, 2*B), dtype=np.complex128)

    # storing output so can listen back to it, remove when testign memory performance
    output = []
    original = []
    
    # process each block individually
    for block in x_blocks:
        block_output, buffer = process_block_up_ols(block, Y_blocks, buffer, fdl)
        output = np.concatenate((output, block_output)).astype(np.float32)
        original = np.concatenate((original, block)).astype(np.float32)

    return output, original

def process_block_up_ols(x, Y_blocks, buffer, fdl):
    B = len(x)
    
    buffer[:B] = buffer[-B:]
    buffer[-B:] = x

    # fft of buffer
    X = np.fft.fft(buffer)
    # X = X[:B+1]
    # update fdl
    for i in range(fdl.shape[0]-1):
        fdl[i] = fdl[i+1]
    fdl[fdl.shape[0]-1] = X

    output = np.zeros((2*B,), dtype=np.complex128)

    for i in range(fdl.shape[0]):
        Z = np.multiply(fdl[fdl.shape[0]-1-i], Y_blocks[i])
        output += Z
    
    out = np.real(np.fft.ifft(output, 2*B))

    return out[-B:], buffer

if __name__ == '__main__':
    # get data and reverb
    data = load_wav("gs/110_B_SoulChords_05_6_SP.wav").astype(np.float32)
    reverb = load_wav("irs/EchoThiefImpulseResponseLibrary/Venues/SteinmanHall.wav").astype(np.float32)
    
    # test overlap add
    t, mem = overlap_add(data, reverb, 1024, True)
    print("{0} s / s".format(t))
    print("{0} KB".format(mem))
