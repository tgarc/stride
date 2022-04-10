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
        else:
            assert hopsize > 0, "hopsize must be a positive number"
        self.blocksize = blocksize
        self.hopsize = hopsize
        self.overlap = self.blocksize - self.hopsize

    def istride(self, blocks, center=False):
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
        else:
            strides = (blocks.strides[0]*self.hopsize,) + blocks.strides[2:]
            array = _as_strided(blocks, shape=shape, strides=strides)

        if center:
            array = array[self.overlap:-self.overlap]

        return array

    def stride(self, x, truncate=True, center=False, **padkwargs):
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
        center : bool, optional
            Add `blocksize - hopsize` samples to the beginning and end of the signal.
        padkwargs : keyword arguments, optional
            If truncate is False or center is True, these kw arguments will be passed to numpy.pad.

        Returns
        -------
        blocks : ndarray
            Strided array.

        '''
        blockshape = x.shape[1:]
        blockstrides = x.strides
        elemsize = int(np.prod(blockshape)) or 1

        nblocks, rem = divmod(x.size - self.overlap*elemsize, self.hopsize*elemsize)
        if nblocks < 0:
            nblocks = 0
            rem = self.blocksize*elemsize - x.size

        padshape = [*((0,0),)*x.ndim]
        if not truncate and rem > 0:
            padshape[0] = (0, self.blocksize - (rem//elemsize))
            nblocks += 1

        if center:
            p0, p1 = padshape[0]
            padshape[0] = (p0+self.overlap, p1+self.overlap)
            nblocks += 2

        if np.any(padshape):
            x = np.pad(x, padshape, **padkwargs)

            # reset strides since this is new memory
            blockstrides = x.strides

        blocks = _as_strided(x, shape=(nblocks, self.blocksize) + blockshape, strides=(self.hopsize*blockstrides[0],) + blockstrides)

        return blocks

    def stride_index(self, x, truncate=True, center=False, fs=1, **padkwargs):
        X = self.stride(x, truncate=truncate, center=center, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        return X, t

    def stridemap(self, func, x, truncate=True, center=False, keepshape=False, keepdims=False, **padkwargs):
        X = self.stride(x, truncate=truncate, center=center, **padkwargs)
        Y = func(X, axis=1, keepdims=keepshape or keepdims)
        if keepshape:
            y = self.istride(Y, center=center)
            if not truncate:
                y = y[:len(x)]
        else:
            y = Y
        return y

    def stridemap_index(self, func, x, truncate=True, center=False, keepshape=False, keepdims=False, fs=1, **padkwargs):
        y = self.stridemap(func, x, truncate=truncate, center=center, keepshape=keepshape, keepdims=keepdims, fs=fs, **padkwargs)
        if keepshape:
            t = np.arange(y.shape[0]) / fs
        else:
            t = np.arange(y.shape[0]) * self.hopsize / fs
        return y, t

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize)

def stride(x, blocksize, hopsize=None, truncate=True, center=False, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stride(x, truncate=truncate, center=center, **kwargs)

def stride_index(x, blocksize, hopsize=None, truncate=True, center=False, fs=1, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stride_index(x, truncate=truncate, center=center, fs=fs, **kwargs)

def istride(X, blocksize, hopsize=None, center=True):
    return Strider(blocksize, hopsize=hopsize).istride(X, center=False)

def stridemap(func, x, blocksize, hopsize=None, truncate=True, center=False, keepshape=False, keepdims=False, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stridemap(func, x, truncate=truncate, center=center, keepshape=keepshape, keepdims=keepdims, **kwargs)

def stridemap_index(func, x, blocksize, hopsize=None, truncate=True, center=False, keepshape=False, keepdims=False, fs=1, **kwargs):
        return Strider(blocksize, hopsize=hopsize).stridemap_index(func, x, truncate=truncate, center=center, keepshape=keepshape, keepdims=keepdims, fs=fs, **kwargs)

class STFTStrider(Strider):

    def __init__(self, window, hopsize=None, nfft=None):
        '''STFTStrider

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
        blocksize = len(window)
        super(STFTStrider, self).__init__(blocksize, hopsize)
        self.window = window

    def stft(self, x, truncate=True, center=False, onesided=None, **padkwargs):
        '''\
        Transform input signal into a tensor of strided (possibly overlapping) windowed 1-D DFTs
        '''
        window = self.window.reshape((1,) + self.window.shape + (1,) * len(x.shape[1:]))

        X = self.stride(x, truncate=truncate, center=center, **padkwargs) * window

        if self.nfft > self.blocksize:
            padshape = [*((0,0),) * X.ndim]
            padshape[1] = (0, self.nfft - self.blocksize)
            X = np.pad(X, padshape, **padkwargs)

        iscomplex = np.iscomplexobj(x)
        if onesided and iscomplex:
            raise ValueError("(onesided=True); Can't take a one-sided fft of a complex input")
        elif onesided == False or iscomplex:
            fft = np.fft.fft
        else:
            fft = np.fft.rfft

        X = fft(X, n=self.nfft, axis=1)

        return X

    def stft_index(self, x, truncate=True, center=False, onesided=None, fs=1, **padkwargs):
        X = self.stft(x, truncate=truncate, center=center, onesided=onesided, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        f = np.arange(X.shape[1]) * fs / self.nfft
        return X, t, f

    def istft(self, X, center=False):
        nblocks = len(X)
        blockshape = X.shape[2:]

        window = self.window.reshape(self.window.shape + (1,) * len(blockshape))
        assert X.ndim > 1, "STFT input should be at least 2-d"

        if X.ndim == 1:
            X = X.reshape((len(X), 1))

        shape = (nblocks * self.hopsize + self.overlap,) + blockshape

        if X.shape[1] == self.nfft//2+1:
            ifft = np.fft.irfft
            dtype = X.real.dtype
        elif X.shape[1] == self.nfft:
            ifft = np.fft.ifft
            dtype = X.dtype
        else:
            assert False, "Unexpected DFT length"

        # Compute per block ifft
        iX = ifft(X, n=self.nfft, axis=1)
        iX = iX[:, :self.blocksize] * window

        # TODO vectorize this
        w2 = window**2
        x = np.zeros(shape, dtype=dtype)
        n = np.zeros(shape, dtype=dtype)
        for i in range(nblocks):
            x[i*self.hopsize:i*self.hopsize+self.blocksize] += iX[i]
            n[i*self.hopsize:i*self.hopsize+self.blocksize] += w2
        x[n != 0] /= n[n != 0]

        if center:
            x = x[self.overlap:-self.overlap]

        return x

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d, nfft=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize, self.nfft)

def stft(x, window, hopsize=None, nfft=None, truncate=True, center=False, onesided=None, **kwargs):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft).stft(x, truncate=truncate, center=center, onesided=onesided, **kwargs)

def stft_index(x, window, hopsize=None, nfft=None, truncate=True, center=False, onesided=None, fs=1,  **kwargs):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft).stft_index(x, truncate=truncate, center=center,onesided=onesided, fs=fs, **kwargs)

def istft(X, window, hopsize=None, nfft=None, center=False):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft).istft(X, center=center)