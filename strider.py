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

    def from_strided(self, blocks, dtype=None, shape=None, fill=0):
        '''Transfrom tensor back to a non-strided version of itself

        Parameters
        ----------
        blocks : ndarray
            _description_
        dtype : dtype, optional
            _description_, by default None
        shape : tuple, optional
            _description_, by default None
        fill : int, optional
            _description_, by default 0

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        '''
        nblocks = len(blocks)
        blockshape = blocks.shape[2:]
        #assert blocks.ndim > 1, "Blocked input should be at least 2-d"

        # import pdb; pdb.set_trace()

        # Assume that if the dimensions have been reduced, a function was applied across the windows
        # in which case from_strided will tile the function output to match the original input signal shape
        # FAILCASE: STFT.from_strided when NFFT > blocksize (i.e. numpy.fft is doing the padding)
        if blocks.ndim == 1:
            blocks = blocks.reshape((len(blocks), 1))
        elif blocks.shape[1] != self.blocksize and blocks.shape[1] != 1:
            blockshape = blocks.shape[1:]
            blocks = blocks.reshape(blocks.shape[:1] + (1,) + blocks.shape[1:])

        if dtype is None:
            dtype = blocks.dtype
        if shape is None:
            shape = (nblocks * self.hopsize + self.overlap,) + blockshape
        elif np.prod(shape) < np.prod((nblocks * self.hopsize + self.overlap,) + blockshape):
            raise ValueError("shape=%s isn't large enough to hold output of shape %s" % (shape, (nblocks * self.hopsize + self.overlap,) + blockshape))

        if blocks.shape[1] == 1:
            """
            This is a trick reserved for reshaping the output of block aggregate
            functions to match the original input signal shape by tiling (i.e.
            repeating) the function output.
            Example:
              strdr = Strider(200, 100)
              wx = strdr.to_strided(x)
              wxdB = 10 * np.log10(np.mean(x**2, axis=1, keepdims=True))
              xdB = strdr.from_strided(wxdB, shape=wx.shape)
            """
            array = np.zeros(shape, dtype=dtype)
            subarry = array[:nblocks*self.hopsize]
            subarry.shape = (nblocks, self.hopsize) + blockshape
            subarry[:nblocks] = blocks # broadcast assign
            array[nblocks*self.hopsize:] = subarry[-1] # fill remainder with edge value
        elif self.overlap == 0 and np.prod(shape) == blocks.size:
            # Just collapse the second dimension back into the first
            array = blocks
            array.shape = (array.shape[0]*array.shape[1],) + blockshape
        else:
            # Make a new array, copying out only the non-overlapping data
            array = np.ones(shape, dtype=dtype)
            array[:nblocks*self.hopsize] = blocks[:nblocks, :self.hopsize, ...].reshape((nblocks*self.hopsize,) + blockshape)
            array[nblocks*self.hopsize:nblocks*self.hopsize+self.overlap] = blocks[nblocks-1, self.hopsize:, ...].reshape((self.overlap,) + blockshape)
            array[nblocks*self.hopsize+self.overlap:] *= fill

        return array

    def to_strided(self, x, pad=False, **padkwargs):
        '''\
        Transforms input signal into a tensor of strided (possibly overlapping) segments

        Parameters
        ----------
        x : ndarray
            input array.
        pad : bool, optional
            Whether to pad the input x so that no samples are dropped. The default is False. !NB! This requires a copy of the input array to be made.
        padkwargs : keyword arguments, optional
            If pad is True, these kw arguments will be passed to numpy.pad.

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
        if pad and rem > 0:
            # import pdb;pdb.set_trace()
            padwidth = self.blocksize - (rem//elemsize)
            padshape = ((0,padwidth),) + ((0,0),)*(x.ndim-1)

            # pad along edge of first dimension
            x = np.pad(x, padshape, **padkwargs)

            # reset strides since this is new memory
            blockstrides = x.strides

            nblocks += 1

        blocks = _as_strided(x, shape=(nblocks, self.blocksize) + blockshape, strides=(self.hopsize*blockstrides[0],) + blockstrides, writeable=writeable)

        return blocks

    def to_strided_index(self, x, pad=False, fs=1, **padkwargs):
        X = self.to_strided(x, pad=pad, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        return X, t

    def stridemap(self, func, x, pad=False, keepshape=False, keepdims=False, **padkwargs):
        if len(padkwargs) == 0:
            padkwargs = dict(mode='reflect')
        X = self.to_strided(x, pad=pad, **padkwargs)
        Y = func(X, axis=1, keepdims=keepshape or keepdims)
        if keepshape:
            y = self.from_strided(Y)
            if pad:
                y = y[:len(x)]
        else:
            y = Y
        return y

    def stridemap_index(self, func, x, pad=False, keepshape=False, keepdims=False, fs=1, **padkwargs):
        y = self.stridemap(func, x, pad=pad, keepshape=keepshape, keepdims=keepdims, **padkwargs)
        if keepshape:
            t = np.arange(y.shape[0]) / fs
        else:
            t = np.arange(y.shape[0]) * self.hopsize / fs
        return y, t

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize)

def to_strided(x, blocksize, hopsize=None, pad=False, **kwargs):
    return Strider(blocksize, hopsize=hopsize).to_strided(x, pad=pad, **kwargs)

def to_strided_index(x, blocksize, hopsize=None, pad=False, fs=1, **kwargs):
    return Strider(blocksize, hopsize=hopsize).to_strided_index(x, pad=pad, fs=fs, **kwargs)

def stridemap(func, x, blocksize, hopsize=None, pad=False, keepshape=False, keepdims=False, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stridemap(func, x, pad=pad, keepshape=keepshape, keepdims=keepdims, **kwargs)

def stridemap_index(func, x, blocksize, hopsize=None, pad=False, keepshape=False, keepdims=False, fs=1, **kwargs):
        return Strider(blocksize, hopsize=hopsize).stridemap_index(func, x, pad=pad, keepshape=keepshape, keepdims=keepdims, fs=fs, **kwargs)

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

    def to_stft(self, x, pad=False, **kwargs):
        '''\
        Transform input signal into a tensor of strided (possibly overlapping) windowed 1-D DFTs
        '''
        window = self.window.reshape((1,) + self.window.shape + (1,) * len(x.shape[1:]))
        X = self.to_strided(x, pad=pad, **kwargs) * window
        return np.fft.rfft(X, n=self.nfft, axis=1)

    def to_stft_index(self, x, pad=False, fs=1, **kwargs):
        X = self.to_stft(x, pad=pad, **kwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        f = np.arange(X.shape[1]) * fs / self.nfft
        return X, t, f

    # def to_lfe(self, x, eps=1e-12, pad=False, **kwargs):
    #     X = self.to_stft(x, pad=pad, **kwargs)
    #     return np.log(1 / self.nfft * np.abs(X)**2 + eps)

    def from_stft(self, X, dtype=float, shape=None):
        nblocks = len(X)
        blockshape = X.shape[2:]
        window = self.window.reshape(self.window.shape + (1,) * len(blockshape))
        #assert X.ndim > 1, "Blocked STFT input should be at least 2-d"

        if X.ndim == 1:
            X = X.reshape((len(X), 1))

        if shape is None:
            shape = (nblocks * self.hopsize + self.overlap,) + blockshape
        elif np.prod(shape) < np.prod((nblocks * self.hopsize + self.overlap,) + blockshape):
            raise ValueError("shape isn't large enough to hold output")

        x = np.zeros(shape, dtype=dtype)
        for i in range(nblocks):
            x[i*self.hopsize:i*self.hopsize+self.blocksize] += np.fft.irfft(X[i], n=self.nfft, axis=0)[:self.blocksize] * window

        return x * self.overlap / np.sum(self.window**2)

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d, nfft=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize, self.nfft)

def to_stft(x, window, hopsize=None, nfft=None, pad=False, **kwargs):
    return RSTFTStrider(window, nfft=nfft, hopsize=hopsize).to_stft(x, pad=pad, **kwargs)

def to_stft_index(x, window, hopsize=None, nfft=None, pad=False, fs=1, **kwargs):
    return RSTFTStrider(window, hopsize=hopsize, nfft=nfft).to_stft_index(x, pad=pad, fs=fs, **kwargs)