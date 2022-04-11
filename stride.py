import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided


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
        center : bool, optional
            Trim `blocksize - hopsize` samples from the beginning and end of the
            signal. (use if center=True was used when calling `stride`).

        Returns
        -------
        x : ndarray
            Un-strided array.
        '''
        nblocks = len(blocks)
        blockshape = blocks.shape[2:]

        # Assume that if the dimensions have been reduced, a function was
        # applied across the windows in which case istride will tile the
        # function output to match the original input signal shape
        # !NB! This function fails for STFT where nfft > blocksize
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
              xdB = strdr.istride(wxdB)
            """
            array = np.zeros_like(blocks, shape=shape)
            subarry = array[:nblocks*self.hopsize]
            subarry.shape = (nblocks, self.hopsize) + blockshape
            subarry[:nblocks] = blocks # broadcast assign
            array[-self.overlap:] = subarry[-1]
        else:
            strides = blocks.strides[1:]
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

def istride(X, blocksize, hopsize=None, center=False):
    return Strider(blocksize, hopsize=hopsize).istride(X, center=center)

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

    def stft(self, x, norm=None, truncate=True, center=False, zerophase=False, **padkwargs):
        '''\
        Transform input signal into a tensor of strided (possibly overlapping) windowed 1-D DFTs
        '''
        # if not fftshift in (None, 'onesided', 'twosided', 'centered'):
        #     raise ValueError("Invalid value for fftshift '%s'" % fftshift)

        window = self.window.reshape((1,) + self.window.shape + (1,) * len(x.shape[1:]))

        iscomplex = np.iscomplexobj(x)
        if iscomplex:
            fft = np.fft.fft
        else:
            fft = np.fft.rfft

        X = self.stride(x, truncate=truncate, center=center, **padkwargs) * window

        # # center the spectrum around f=0
        # if iscomplex and fftshift == 'centered':
        #     X *= np.exp(1j*np.pi*np.arange(self.blocksize))

        if self.nfft > self.blocksize:
            padshape = [*((0,0),) * X.ndim]
            padshape[1] = (0, self.nfft - self.blocksize)
            X = np.pad(X, padshape, **padkwargs)

        if zerophase:
            mod = self.blocksize % 2
            X = np.roll(X, -(self.blocksize-mod)//2, axis=1)

        X = fft(X, n=self.nfft, axis=1, norm=norm)

        # if iscomplex and fftshift == 'onesided':
        #     mod = self.nfft % 2
        #     X[(self.nfft+mod)//2:] += X[(self.nfft-mod)//2::-1]
        #     X = X[:(self.nfft+mod)//2 + 1]

        return X

    def stft_index(self, x, norm=None, truncate=True, center=False, zerophase=False, fs=1, **padkwargs):
        X = self.stft(x, norm=norm, truncate=truncate, center=center, zerophase=zerophase, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        f = np.arange(X.shape[1]) * fs / self.nfft

        # iscomplex = X.shape[1] == self.nfft
        # if iscomplex:
        #     if fftshift == 'centered':
        #         # center the spectrum around f=0
        #         f -= fs/2
        #     elif fftshift == 'onesided':
        #         f = f[(self.nfft+mod)//2:]
        #     elif fftshift == 'twosided' or fftshift is None:
        #         mod = self.nfft % 2
        #         f[(self.nfft+mod)//2:] = -f[(self.nfft-mod)//2::-1]

        return X, t, f

    def istft(self, X, norm=None, center=False, zerophase=False):
        nblocks = len(X)
        blockshape = X.shape[2:]

        window = self.window.reshape(self.window.shape + (1,) * len(blockshape))

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
        iX = ifft(X, n=self.nfft, axis=1, norm=norm)
        if zerophase:
            mod = self.blocksize % 2
            iX = np.roll(iX, (self.blocksize-mod)//2, axis=1)
        iX = iX[:, :self.blocksize] * window

        # # Un-center spectrum
        # if X.shape[1] == self.nfft and fftshift == 'centered':
        #     iX *= np.exp(-1j*np.pi*np.arange(self.blocksize))

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

def stft(x, window, hopsize=None, nfft=None, norm=None, truncate=True, center=False, zerophase=False, **kwargs):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft).stft(x, norm=norm, truncate=truncate, center=center, zerophase=zerophase, **kwargs)

def stft_index(x, window, hopsize=None, nfft=None, norm=None, truncate=True, center=False, zerophase=False, fs=1,  **kwargs):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft).stft_index(x, norm=norm, truncate=truncate, center=center, zerophase=zerophase, fs=fs, **kwargs)

def istft(X, window, hopsize=None, nfft=None, norm=None, center=False, zerophase=False):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft).istft(X, norm=norm, center=center, zerophase=zerophase)