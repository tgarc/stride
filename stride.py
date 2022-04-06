import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided


# Assumes blocks are split into evenly divided parts
# Accepts multi-dimensional input, but 1-d and 2-d are the only ones tested
class Strider(object):

    def __init__(self, blocksize, hopsize=None):
        '''Strider

        Parameters
        ----------
        blocksize : int
            Number of samples in each window.
        hopsize : int, optional
            Size of stride between windows.

        '''
        assert blocksize > 1, "blocksize of 1 not supported"
        if hopsize is None:
            hopsize = blocksize
        self.blocksize = blocksize
        self.hopsize = hopsize
        self.overlap = self.blocksize - self.hopsize

    def istride(self, blocks):
        '''Transfrom tensor back to a non-strided version of itself

        Parameters
        ----------
        blocks : ndarray
            Strided input array.

        Returns
        -------
        x : ndarray
            Un-strided array.
        '''
        nblocks = len(blocks)
        blockshape = blocks.shape[2:]
        #assert blocks.ndim > 1, "Blocked input should be at least 2-d"

        # Assume that if the dimensions have been reduced, a function was applied across the windows
        # in which case istride will tile the function output to match the original input signal shape
        # FAILCASE: STFT.istride when NFFT > blocksize (i.e. numpy.fft is doing the padding)
        # import pdb; pdb.set_trace()
        if blocks.ndim == 1:
            blocks = blocks.reshape((len(blocks), 1))
        elif blocks.shape[1] != self.blocksize and blocks.shape[1] != 1:
            blockshape = blocks.shape[1:]
            blocks = blocks.reshape(blocks.shape[:1] + (1,) + blocks.shape[1:])

        shape = (nblocks * self.hopsize + self.overlap,) + blockshape

        if blocks.shape[1] == 1:
            """
            This is a trick reserved for reshaping the output of block aggregate
            functions to match the original input signal shape by tiling (i.e.
            repeating) the function output.
            Example:
              strdr = Strider(200, 100)
              wx = strdr.stride(x)
              wxdB = 10 * np.log10(np.mean(x**2, axis=1, keepdims=True))
              xdB = strdr.istride(wxdB, shape=wx.shape)
            """
            array = np.zeros(shape, dtype=blocks.dtype)
            subarry = array[:nblocks*self.hopsize]
            subarry.shape = (nblocks, self.hopsize) + blockshape
            subarry[:nblocks] = blocks # broadcast assign
            array[-self.overlap:] = subarry[-1] # fill remainder with edge value
        elif self.overlap == 0 and np.prod(shape) == blocks.size:
            # Just collapse the second dimension back into the first
            array = blocks
            array.shape = (array.shape[0]*array.shape[1],) + blockshape
        else:
            # Make a new array, copying out only the non-overlapping data
            array = np.ones(shape, dtype=blocks.dtype)
            array[:nblocks*self.hopsize] = blocks[:nblocks, :self.hopsize].reshape((nblocks*self.hopsize,) + blockshape)
            array[-self.overlap:] = blocks[nblocks-1, self.hopsize:].reshape((self.overlap,) + blockshape)

        return array

    def stride(self, x, truncate=True, pad_mode='constant', **padkwargs):
        '''\
        Transforms input signal into a tensor of strided (possibly overlapping)
        segments

        Parameters
        ----------
        x : ndarray
            input array.
        truncate : bool, optional
            Truncate remainder samples from input that don't fit the strides
            exactly. If False, the input x will be padded so that no samples are
            dropped; !NB! This requires a copy of the input array to be made.
        padkwargs : keyword arguments, optional
            If truncate is False, these kw arguments will be passed to numpy.pad.

        Returns
        -------
        blocks : ndarray
            Strided array.

        '''
        writeable = (self.overlap == 0) # Only allow writing for non-overlapping strides

        blockshape = x.shape[1:]
        blockstrides = x.strides
        elemsize = int(np.prod(blockshape)) or 1

        nblocks, rem = divmod(x.size - self.overlap*elemsize, self.hopsize*elemsize)
        if nblocks < 0:
            nblocks = 0
            rem = self.blocksize*elemsize - x.size
        if not truncate and rem > 0:
            # import pdb;pdb.set_trace()
            padwidth = self.blocksize - (rem//elemsize)
            padshape = ((0,padwidth),) + ((0,0),)*(x.ndim-1)

            # pad along edge of first dimension
            x = np.pad(x, padshape, mode=pad_mode, **padkwargs)

            # reset strides since this is new memory
            blockstrides = x.strides

            nblocks += 1

        blocks = _as_strided(x, shape=(nblocks, self.blocksize) + blockshape, strides=(self.hopsize*blockstrides[0],) + blockstrides, writeable=writeable)

        return blocks

    def stride_index(self, x, truncate=True, pad_mode='constant', fs=1, **padkwargs):
        X = self.stride(x, truncate=truncate, pad_mode=pad_mode, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        return X, t

    def stridemap(self, func, x, truncate=True, pad_mode='constant', keepshape=False, keepdims=False, **padkwargs):
        X = self.stride(x, truncate=truncate, pad_mode=pad_mode, **padkwargs)
        Y = func(X, axis=1, keepdims=keepshape or keepdims)
        if keepshape:
            y = self.istride(Y)
            if not truncate:
                y = y[:len(x)]
        else:
            y = Y
        return y

    def stridemap_index(self, func, x, truncate=True, pad_mode='constant', keepshape=False, keepdims=False, fs=1, **padkwargs):
        y = self.stridemap(func, x, truncate=truncate, pad_mode=pad_mode, keepshape=keepshape, keepdims=keepdims, fs=fs, **padkwargs)
        if keepshape:
            t = np.arange(y.shape[0]) / fs
        else:
            t = np.arange(y.shape[0]) * self.hopsize / fs
        return y, t

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize)

def stride(x, blocksize, hopsize=None, truncate=True, pad_mode='constant', **kwargs):
    return Strider(blocksize, hopsize=hopsize).stride(x, truncate=truncate, pad_mode=pad_mode, **kwargs)

def stride_index(x, blocksize, hopsize=None, truncate=True, pad_mode='constant', fs=1, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stride_index(x, truncate=truncate, pad_mode=pad_mode, fs=fs, **kwargs)

def istride(X, blocksize, hopsize=None):
    return Strider(blocksize, hopsize=hopsize).stride(X)

def stridemap(func, x, blocksize, hopsize=None, truncate=True, pad_mode='constant', keepshape=False, keepdims=False, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stridemap(func, x, truncate=truncate, pad_mode=pad_mode, keepshape=keepshape, keepdims=keepdims, **kwargs)

def stridemap_index(func, x, blocksize, hopsize=None, truncate=True, pad_mode='constant', keepshape=False, keepdims=False, fs=1, **kwargs):
        return Strider(blocksize, hopsize=hopsize).stridemap_index(func, x, truncate=truncate, pad_mode=pad_mode, keepshape=keepshape, keepdims=keepdims, fs=fs, **kwargs)

class RSTFTStrider(Strider):

    def __init__(self, window, hopsize=None, nfft=None):
        '''RSTFTStrider

        Parameters
        ----------
        window : ndarray or scalar
            (ndarray) Pre-fft window to apply
            (scalar) Number of sample points to use per FFT. In this case no windowing will be applied before FFT.
        hopsize : int, optional
            Number of samples to skip between windows
        nfft : int, optional
            FFT size (should be >= window size to avoid truncation). The default sets the FFT size equal to the window size.

        Returns
        -------
        None.

        '''
        if np.isscalar(window):
            window = np.ones(window)
        self.nfft = nfft or len(window)
        super(RSTFTStrider, self).__init__(len(window), hopsize)
        self.window = window

    def stft(self, x, truncate=True, pad_mode='constant', **padkwargs):
        '''\
        Transform input signal into a tensor of strided (possibly overlapping) windowed 1-D DFTs
        '''
        window = self.window.reshape((1,) + self.window.shape + (1,) * len(x.shape[1:]))
        X = self.stride(x, truncate=truncate, pad_mode=pad_mode, **padkwargs) * window
        return np.fft.rfft(X, n=self.nfft, axis=1)

    def stft_index(self, x, truncate=True, pad_mode='constant', fs=1, **padkwargs):
        X = self.stft(x, truncate=truncate, pad_mode=pad_mode, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        f = np.arange(X.shape[1]) * fs / self.nfft
        return X, t, f

    def istft(self, X):
        nblocks = len(X)
        blockshape = X.shape[2:]
        window = self.window.reshape(self.window.shape + (1,) * len(blockshape))
        #assert X.ndim > 1, "Blocked STFT input should be at least 2-d"

        if X.ndim == 1:
            X = X.reshape((len(X), 1))

        shape = (nblocks * self.hopsize + self.overlap,) + blockshape

        # TODO vectorize this
        x = np.zeros(shape, dtype=X.real.dtype)
        n = np.zeros(shape, dtype=X.real.dtype)
        w2 = window**2
        iX = np.fft.irfft(X, n=self.nfft, axis=1)[:, :self.blocksize] * window
        for i in range(nblocks):
            x[i*self.hopsize:i*self.hopsize+self.blocksize] += iX[i]
            n[i*self.hopsize:i*self.hopsize+self.blocksize] += w2

        x[n != 0] /= n[n != 0]
        return x

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d, nfft=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize, self.nfft)

def stft(x, window, hopsize=None, nfft=None, truncate=True, pad_mode='constant', **kwargs):
    return RSTFTStrider(window, hopsize=hopsize, nfft=nfft).stft(x, truncate=truncate, pad_mode=pad_mode, **kwargs)

def stft_index(x, window, hopsize=None, nfft=None, truncate=True, pad_mode='constant', fs=1,  **kwargs):
    return RSTFTStrider(window, hopsize=hopsize, nfft=nfft).stft_index(x, truncate=truncate, pad_mode=pad_mode, fs=fs, **kwargs)

def istft(X, window, hopsize=None, nfft=None):
    return RSTFTStrider(window, hopsize=hopsize, nfft=nfft).istft(X)