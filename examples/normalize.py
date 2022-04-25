'''
Z-Normalize a signal based on short-time statistics
'''
# %% Setup
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import stride as st

# signal sampling rate
fs = 16000

# Input signal
x = np.random.randn(fs * 5)

# Analysis intervals in ms
blocksize = 0.016
hopsize = 0.008

# %% Normalize
L = int(blocksize*fs)
K = int(hopsize*fs)

# Segment signal, padding the end of the signal so no truncation occurs
X = st.stride(x, L, K, truncate=False, mode='reflect')

# Calculate local statistics
# Retain the singleton dimension after operations - this will allow using broadcast operations
mu = np.mean(X, axis=1, keepdims=True)
std = np.std(X, axis=1, keepdims=True)

# Normalize
X_norm = (X - mu) / std