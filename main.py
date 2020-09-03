import cv2
import matplotlib.pyplot as plt
import functions as fnc
import numpy as np
from scipy import interpolate
from scipy import signal
from scipy.optimize import curve_fit
from scipy.fftpack import fft, ifft, fftfreq

# Load and Display Image #
im_loc = fnc.dict['fname']
im = cv2.imread(im_loc, 0)
fig, ax = plt.subplots()
img = ax.imshow(im)
plt.show()
# Compute Histogram and normalize #
plt.figure()
counts, bins, bars = plt.hist(im.ravel(), bins=255, range=(0.0, 255.0), fc='k', ec='k')
plt.close()
y = counts/im.size

# Plot Histogram #
plt.figure(figsize=(10, 5))
xbins = bins[:-1]+np.diff(bins, axis=0)/2
plt.plot(xbins, y, 'k-')
plt.title("Normalized histogram")
plt.show()

# Interpolate Function #
f = interpolate.interp1d(xbins, y, kind="cubic")
xnew = np.arange(0.5, 254, 0.1)
ynew = f(xnew)

# Find Peaks After Interpolation#
signal.find_peaks(xnew, ynew)
peaks, _ = signal.find_peaks(ynew, height=(0, 1))
plt.figure(figsize=(10, 5))
plt.plot(xnew, ynew, 'k-')
plt.plot(xnew[peaks], ynew[peaks], 'rx')
plt.title("Upsampled and interpolated signal with noisy peaks")
plt.show()

# Now make a new Band-pass Filter (low pass, in this case).
# To do this, go from real space to Fourier space and remove
# some higher frequencies.

# First, lets compute our spatial resolution (how close our domain points are to each other).
dx = xnew[1]-xnew[0]

# Compute the FFT.  It is convention to use upper-case letters to indicate Fourier transformed variables #
YNEW = fft(ynew)

# Compute Magnitude of the FFT #
power = np.abs(YNEW)

# The corresponding frequencies to be plotted on the x-axis.  In the original signal, we had pixel intensity
# on the x-axis (the bins of the histogram).  In Fourier space, we have the frequency on the x-axis.  So this function
# uses the original x-axis and the size of the fourier transform array to output the new x-axis of frequencies.  #
freqs = fftfreq(YNEW.size, d=dx)

# Only keep positive frequencies (what does a negative frequency even mean?) #
nnf = np.where(freqs > 0)
freqsh = freqs[nnf]
power = power[nnf]

# Plot the FFT power
plt.figure(figsize=(10, 5))
plt.plot(freqsh, power)
plt.xlabel('Frequency')
plt.ylabel('Magnitude(YNEW)')
plt.title("Plot of frequency spectrum our signal")
plt.show()

# Here is where we actually (low-pass) filter our original signal.  The value of 0.1 is arbitrary, so you can
# try to find a value that works better to identify the peaks in the signal.
YNEWCopy = YNEW.copy()
YNEWCopy[np.abs(freqs) > 0.1] = 0

# Now we transform back from Fourier space into real space, so we go back to using lower-case letters for our
# variables.  This ynewfilt is now a low-pass filtered version of the original ynew variable.
ynewfilt = ifft(YNEWCopy)
ynewfilt = np.real(ynewfilt)

# Find Peaks with filtered signal #
signal.find_peaks(xnew, ynewfilt)
peaksfilt, _ = signal.find_peaks(ynewfilt, height=(0.001, 1))


## Perform a Gaussian Fit (to the last peak in the data ##
popt, pcov = curve_fit(fnc.func, xnew[peaksfilt[-1]:], ynewfilt[peaksfilt[-1]:])

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(xnew, ynew, label='Original')
plt.plot(xnew, ynewfilt, linewidth=2, label='Filtered')
plt.plot(xnew[peaksfilt], ynewfilt[peaksfilt], 'rx', label='Peaks')
plt.xlabel('Intensity')
plt.ylabel('Amplitude')
plt.title("Filtered and Upsampled Signal with clean peaks")
plt.legend(loc='best')

## Add Plot of fitted gaussian ##
ym = fnc.func(xnew, popt[0], popt[1])
ax.plot(xnew, ym, c='r', label='Gaussian fit')
ax.legend()

plt.show()