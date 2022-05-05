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
            assert 0 < hopsize <= blocksize, "hopsize must be a positive integer between 0 and blocksize"

        self.blocksize = blocksize
        self.hopsize = hopsize
        self.overlap = self.blocksize - self.hopsize

    def istride(self, blocks, edgepadded=False):
        '''Transfrom tensor back to a non-strided version of itself

        Parameters
        ----------
        blocks : ndarray
            Strided input array.
        edgepadded : bool, optional
            If True, trim `blocksize - hopsize` samples from the beginning and
            end of the signal. (use if edgepadded=True was used when calling
            `stride`).

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
            array[-self.overlap:] = blocks[-1]
        else:
            strides = blocks.strides[1:]
            array = _as_strided(blocks, shape=shape, strides=strides)

        if edgepadded:
            array = array[self.overlap:-self.overlap]

        return array

    def stride(self, x, truncate=True, edgepadded=False, **padkwargs):
        '''Transforms input signal into a tensor of strided (possibly
        overlapping) segments

        Parameters
        ----------
        x : ndarray
            input array.
        truncate : bool, optional
            Truncate remainder samples from input that don't fit the strides
            exactly. If `False`, the input array will be padded so that no
            samples are dropped.
        edgepadded : bool, optional
            Add `blocksize - hopsize` samples to the beginning and end of the
            input array. Additional padding may be added to the right edge of
            the array to avoid dropping samples.
        padkwargs : keyword arguments, optional
            If `truncate` is False or `edgepadded` is True, these kw arguments
            will be passed to `numpy.pad`.

        Returns
        -------
        blocks : ndarray
            Strided array.

        '''
        blockshape = x.shape[1:]
        blockstrides = x.strides
        elemsize = int(np.prod(blockshape)) or 1
        n = x.size // elemsize

        nblocks, rem = divmod(n - self.overlap, self.hopsize)
        if nblocks < 0:
            nblocks = 0
            rem = self.blocksize - n

        padshape = [*((0,0),)*x.ndim]
        if not truncate and rem > 0:
            padshape[0] = (0, self.blocksize - rem)
            nblocks += 1

        if edgepadded:
            p0, p1 = padshape[0]
            lpad = p0 + self.overlap
            rpad = p1

            nblocks, rem = divmod(n + lpad + rpad, self.hopsize)
            if rem > 0:
                rpad += self.blocksize - rem
                nblocks += 1
            else:
                rpad += self.overlap
            padshape[0] = (lpad, rpad)

        # print(nblocks*self.hopsize + self.overlap)
        if np.any(padshape):
            x = np.pad(x, padshape, **padkwargs)

            # reset strides since this is new memory
            blockstrides = x.strides
        # print(x.size//elemsize)

        blocks = _as_strided(x, shape=(nblocks, self.blocksize) + blockshape, strides=(self.hopsize*blockstrides[0],) + blockstrides)

        return blocks

    def stride_index(self, x, truncate=True, edgepadded=False, fs=1, **padkwargs):
        X = self.stride(x, truncate=truncate, edgepadded=edgepadded, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        return X, t

    def istride_index(self, blocks, edgepadded=False, fs=1):
        x = self.istride(blocks, edgepadded=edgepadded)
        t = np.arange(x.shape[0]) / fs
        return x, t

    def stridemap(self, ufunc, x, truncate=True, edgepadded=False, keepdims=False, **padkwargs):
        X = self.stride(x, truncate=truncate, edgepadded=edgepadded, **padkwargs)
        return ufunc(X, axis=1, keepdims=keepdims)

    def stridemap_index(self, ufunc, x, truncate=True, edgepadded=False, keepdims=False, fs=1, **padkwargs):
        y = self.stridemap(ufunc, x, truncate=truncate, edgepadded=edgepadded, keepdims=keepdims, fs=fs, **padkwargs)
        t = np.arange(y.shape[0]) * self.hopsize / fs
        return y, t

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize)

def stride(x, blocksize, hopsize=None, truncate=True, edgepadded=False, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stride(x, truncate=truncate, edgepadded=edgepadded, **kwargs)

def stride_index(x, blocksize, hopsize=None, truncate=True, edgepadded=False, fs=1, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stride_index(x, truncate=truncate, edgepadded=edgepadded, fs=fs, **kwargs)

def istride(X, blocksize, hopsize=None, edgepadded=False):
    return Strider(blocksize, hopsize=hopsize).istride(X, edgepadded=edgepadded)

def istride_index(blocks, blocksize, hopsize=None, edgepadded=False):
    return Strider(blocksize, hopsize=hopsize).istride_index(blocks, edgepadded=edgepadded)

def stridemap(ufunc, x, blocksize, hopsize=None, truncate=True, edgepadded=False, keepdims=False, **kwargs):
    return Strider(blocksize, hopsize=hopsize).stridemap(ufunc, x, truncate=truncate, edgepadded=edgepadded, keepdims=keepdims, **kwargs)

def stridemap_index(ufunc, x, blocksize, hopsize=None, truncate=True, edgepadded=False, keepdims=False, fs=1, **kwargs):
        return Strider(blocksize, hopsize=hopsize).stridemap_index(ufunc, x, truncate=truncate, edgepadded=edgepadded, keepdims=keepdims, fs=fs, **kwargs)

class STFTStrider(Strider):

    def __init__(self, window, hopsize=None, nfft=None, centeredfft=False, zerophase=False, norm=None):
        '''STFTStrider

        Parameters
        ----------
        window : ndarray or scalar
            (ndarray)   Pre-fft window to apply
            (scalar)    Number of sample points to use per FFT. In this case no
                        windowing will be applied before FFT.
        hopsize : int, optional
            Number of samples to skip between windows
        nfft : int, optional
            FFT size (should be >= window size to avoid truncation). The default
            sets the FFT size equal to the window size.
        centeredfft : bool, optional
            Center the spectrum.
        zerophase : bool, optional
            Eliminate phase skew when applying DFT.
        norm : -
            Which FFT normalization factor to apply. See `numpy.fft.fft`
            documentation.

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
        self.centeredfft = centeredfft
        self.zerophase = zerophase
        self.norm = norm

    def stft(self, x, truncate=True, edgepadded=False, **padkwargs):
        '''Transform input signal into a tensor of strided (possibly
        overlapping) windowed 1-D DFTs

        Parameters
        ----------
        truncate : bool or None, optional
            Truncate remainder samples from input that don't fit the strides
            exactly. If `False`, the input x will be padded so that no samples
            are dropped.
        edgepadded : bool, optional
            Add `blocksize - hopsize` samples to the beginning and end of the
            signal. Adding this padding makes it possible to perfectly
            reconstruct the input at the edges of the signal (assuming other
            requisite conditions are met).
        '''
        window = self.window.reshape((1,) + self.window.shape + (1,) * len(x.shape[1:]))

        iscomplex = np.iscomplexobj(x)
        if iscomplex:
            fft = np.fft.fft
        else:
            fft = np.fft.rfft

        X = self.stride(x, truncate=truncate, edgepadded=edgepadded, **padkwargs) * window

        # center the spectrum around f=0
        if iscomplex and self.centeredfft:
            X *= np.exp(1j*np.pi*np.arange(self.blocksize))
        # elif self.centeredfft:
        #     warnings.warn("centeredfft=True is only valid when input is complex")

        if self.nfft > self.blocksize:
            padshape = [*((0,0),) * X.ndim]
            padshape[1] = (0, self.nfft - self.blocksize)
            X = np.pad(X, padshape, **padkwargs)

        if self.zerophase:
            mod = self.blocksize % 2
            X = np.roll(X, -(self.blocksize-mod)//2, axis=1)

        X = fft(X, n=self.nfft, axis=1, norm=self.norm)

        return X

    def stft_index(self, x, truncate=True, edgepadded=False, fs=1, **padkwargs):
        X = self.stft(x, truncate=truncate, edgepadded=edgepadded, **padkwargs)
        t = np.arange(X.shape[0]) * self.hopsize / fs
        f = np.arange(X.shape[1]) * fs / self.nfft

        if X.shape[1] == self.nfft:
            if self.centeredfft:
                # center the spectrum around f=0
                f -= fs/2
            else:
                mod = self.nfft % 2
                f[self.nfft//2+1-mod:] = -f[self.nfft//2-1+mod:0:-1]

        return X, t, f

    def istft(self, X, edgepadded=False):
        '''Invert stft transform.

        Parameters
        ----------
        edgepadded : bool, optional
            If True, trim `blocksize - hopsize` samples from the beginning and
            end of the signal. (use if edgepadded=True was used when calling
            `stft`).
        '''
        nblocks = len(X)
        blockshape = X.shape[2:]

        window = self.window.reshape(self.window.shape + (1,) * len(blockshape))

        if X.ndim == 1:
            X = X.reshape((len(X), 1))

        shape = (nblocks * self.hopsize + self.overlap,) + blockshape

        iscomplex = X.shape[1] == self.nfft
        if iscomplex:
            ifft = np.fft.ifft
            dtype = X.dtype
        elif X.shape[1] == self.nfft//2+1:
            ifft = np.fft.irfft
            dtype = X.real.dtype
        else:
            raise ValueError("Unexpected DFT length")

        # Compute per block ifft
        iX = ifft(X, n=self.nfft, axis=1, norm=self.norm)
        if self.zerophase:
            mod = self.blocksize % 2
            iX = np.roll(iX, (self.blocksize-mod)//2, axis=1)
        iX = iX[:, :self.blocksize] * window

        # Un-center spectrum
        if iscomplex and self.centeredfft:
            iX *= np.exp(-1j*np.pi*np.arange(self.blocksize))

        # TODO vectorize this
        w2 = window**2
        x = np.zeros(shape, dtype=dtype)
        n = np.zeros(shape, dtype=dtype)
        for i in range(nblocks):
            x[i*self.hopsize:i*self.hopsize+self.blocksize] += iX[i]
            n[i*self.hopsize:i*self.hopsize+self.blocksize] += w2
        mask = n != 0
        x[mask] /= n[mask]

        if edgepadded:
            x = x[self.overlap:-self.overlap]

        return x

    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d, nfft=%d, centeredfft=%s, zerophase=%s, norm=%s)" % (self.__class__.__name__, self.blocksize, self.hopsize, self.nfft, self.centeredfft, self.zerophase, self.norm)

def stft(x, window, hopsize=None, nfft=None, truncate=True, edgepadded=False, centeredfft=False, zerophase=False, norm=None, **kwargs):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft, centeredfft=centeredfft, zerophase=zerophase, norm=norm).stft(x, truncate=truncate, edgepadded=edgepadded, **kwargs)

def stft_index(x, window, hopsize=None, nfft=None, truncate=True, edgepadded=False, centeredfft=False, zerophase=False, norm=None, fs=1, **kwargs):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft, centeredfft=centeredfft, zerophase=zerophase, norm=norm).stft_index(x, truncate=truncate, edgepadded=edgepadded, fs=fs, **kwargs)

def istft(X, window, hopsize=None, nfft=None, edgepadded=False, centeredfft=False, zerophase=False, norm=None):
    return STFTStrider(window, hopsize=hopsize, nfft=nfft, centeredfft=centeredfft, zerophase=zerophase, norm=norm).istft(X, edgepadded=edgepadded)