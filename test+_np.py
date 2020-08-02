import numpy as np
from matplotlib.pyplot import plot, show
# import matplotlib.pyplot as plt
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plot(freq, sp.real, freq, sp.imag)
show()