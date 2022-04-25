'''
Mel-frequency Cepstral Coefficients feature extraction
'''
#%% Setup
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import stride as st
import soundfile as sf
import matplotlib.pyplot as plt

# signal sampling rate
fs = 16000

# Input signal
x, fs = sf.read('../data/sa1.flac')

# Analysis intervals in ms
blocksize = 0.016
hopsize = 0.008

# convert to samples
L = int(fs*blocksize)
K = int(fs*hopsize)

# MFCC parameters
window = np.sqrt(np.hanning(L+1)[1:])   # Analysis window
nfft = 2048                             # FFT size
M = 64                                  # Number of Mel filters
eps = 1e-12                             # Regularization constant

#%% Helper functions
def freq2mel(f):
    # From O'Shaughnessy, Speech Communication
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel2freq(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def createfilter(M, nfft, fs=1):
    nbins = nfft//2 + 1
    H = np.zeros((M, nbins))
    binwidth = float(fs) / nfft

    mels = np.linspace(0, freq2mel(fs/2), M+2)
    freqs = mel2freq(mels)
    bins = (freqs * (nfft+1) / fs).astype(int)
    freqs = bins * binwidth

    for m in range(M):
        l,c,r = bins[m:m+3]
        H[m, l:c] = (np.arange(l,c) - l) / (c - l)
        H[m, c:r+1] = (r - np.arange(c,r+1)) / (r - c)

    return H, freqs[1:-1]

# %% Log Frequency Power features

# Apply STFT and retrive time/frequency indices
X, t, f = st.stft_index(x, window, K, nfft, fs=fs)

# Power spectrum
Xpow = 1/nfft * np.abs(X)**2

# Log Frequency Power
X_lfe = np.log(Xpow + eps)

# %% Log Mel-frequency Power features
# Generate Mel filter bank
H, fm = createfilter(M, nfft, fs)

# Apply filter bank to get Mel power spectrum
Xpow_mel = Xpow @ H.T

# Mel frequency power
X_mfe = np.log(Xpow_mel + eps)

# %% Mel-frequency Cepstral Coefficients
from scipy.fftpack import dct

# Apply DCT to get cepstral coefficients
X_mfcc = dct(X_mfe, norm='ortho')

# %% Plot Log Frequency Power
plt.pcolormesh(t, f, X_lfe.T, shading='auto')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Log Frequency Power')

# %% Plot Log Mel-frequency Power
plt.pcolormesh(t, fm, X_mfe.T, shading='auto')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Log Mel-frequency Power (M=%d)' % M)
plt.gca().set_yscale("function", functions=[freq2mel, mel2freq])

# %% Mel-frequency Cepstral Coefficients
# (Drop DC coefficient for plotting)
plt.pcolormesh(t, np.arange(1, M), X_mfcc[:,1:].T, shading='auto', rasterized=True, edgecolors='none')
plt.xlabel('Time (s)')
plt.title('Mel-frequency Cepstral Coefficients (M=%d)' % M)

# %%
