# Overview
A minimalist toolset for applying block-based signal analysis based
on the numpy library.


The purpose of this library is to enable the development of block based signal
analysis in a compact and efficient (vector based) way. By using the stride
properties underlying numpy arrays, these tools allow creating views of
(possibly overlapping) arrays without allocating additional memory.

# Installation

`pip install -r requirements.txt`

# Dependencies
* Python 3.6 or greater
* See `requirements.txt`

# Notes

* Striding is always done along the first axis of an input array
* The first two dimensions of a strided array will always be of shape `(number of
  blocks, blocksize)`; the remaining dimensions will match the input signal
  dimensions excluding the first axis
  ```python
  x = np.ones((120, 5, 7, 2))
  X = st.stride(x, 10, 10)
  # Split array into 120/10 == 12 blocks of size 10
  # > X.shape[:2]
  # (12, 10)
  # Remaining dimensions are unchanged
  # > X.shape[2:]
  # (5,7,2)
  ```

# Examples

## Initial setup
```python
# Setup
import numpy as np
import stride as st

fs = 16000
blocksize = 0.016
hopsize = 0.008

x = np.random.randn(fs * 5)

# Segment signals into `L` sized blocks with a hopsize of `K`
L = int(blocksize*fs)
K = int(hopsize*fs)

strider = st.Strider(L, K)
# > strider
# Strider(blocksize=256, hopsize=128)
```
### Calculate Short-time RMS Signal Level
```python
# Segment signal
X = st.stride(x, L, K)
# > X.shape
# (624, 256)

# Apply rms metric over windows (windows are always along axis=1)
xdB = 10 * np.log10(np.mean(X**2, axis=1))
# > xdB.shape
# (624,)
```

### Get Normalized Signal Segments

```python
X = st.stride(x, L, K)

# Get local statistics
# (Retain the singleton dimension - this will allow using broadcast operations)
mu = np.mean(X, axis=1, keepdims=True)
std = np.std(X, axis=1, keepdims=True)

# Normalize
X_norm = (X - mu) / std
```

### Apply Frequency Domain Processing
```python
# Get the next power of 2 size for FFT processing
nfft = max(512, 1 << (int(np.ceil(np.log2(L)))))

# Define the tapering window
w = np.hamming(L+1)[1:]

strider = st.STFTStrider(w, K, nfft)
# > strider
# STFTStrider(blocksize=256, hopsize=128, nfft=512)

X = strider.stft(x)

# Apply processing on X ...
Y = X

# Reconstruct the time-domain signal
y = strider.istft(Y)
```
