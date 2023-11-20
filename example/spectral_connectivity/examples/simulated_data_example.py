import numpy as np
import matplotlib.pyplot as plt
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity
from spectral_connectivity import multitaper_connectivity


####  Test Power Spectrum 200 Hz signal
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 50)
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signal = np.sin(2 * np.pi * time * frequency_of_interest)
noise = np.random.normal(0, 4, len(signal))

plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.plot(time, signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal', fontweight='bold')
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))

plt.subplot(2, 2, 2)
plt.plot(time, signal + noise)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal + Noise', fontweight='bold')
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))

plt.subplot(2, 2, 3)
m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
plt.plot(c.frequencies, c.power().squeeze())
plt.subplot(2, 2, 4)
m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)

plt.plot(c.frequencies, c.power().squeeze())


####  30 Hz signal
frequency_of_interest = 30
sampling_frequency = 1500
time_extent = (0, 50)
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signal = np.sin(2 * np.pi * time * frequency_of_interest)
noise = np.random.normal(0, 4, len(signal))

plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.plot(time, signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal', fontweight='bold')
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))

plt.subplot(2, 2, 2)
plt.plot(time, signal + noise)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal + Noise', fontweight='bold')
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))

plt.subplot(2, 2, 3)
m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
plt.plot(c.frequencies, c.power().squeeze());
plt.subplot(2, 2, 4)
m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)

plt.plot(c.frequencies, c.power().squeeze())


#### Spectrogram:  No trials, 200 Hz signal with 50 Hz signal starting at 25 seconds

frequency_of_interest = [200, 50]
sampling_frequency = 1500
time_extent = (0, 50)
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
signal[:n_time_samples // 2, 1] = 0
signal = signal.sum(axis=1)
noise = np.random.normal(0, 4, signal.shape)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
axes[0, 0].plot(time, signal)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Signal', fontweight='bold')
axes[0, 0].set_xlim((24.90, 25.10))
axes[0, 0].set_ylim((-10, 10))

axes[0, 1].plot(time, signal + noise)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_title('Signal + Noise', fontweight='bold')
axes[0, 1].set_xlim((24.90, 25.10))
axes[0, 1].set_ylim((-10, 10))

m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
axes[1, 0].plot(c.frequencies, c.power().squeeze())

m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
axes[1, 1].plot(c.frequencies, c.power().squeeze())


m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               time_window_duration=0.600,
               time_window_step=0.300,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)
mesh = axes[2, 0].pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
axes[2, 0].set_ylim((0, 300))
axes[2, 0].axvline(time[int(np.fix(n_time_samples / 2))], color='black')

m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               time_window_duration=0.600,
               time_window_step=0.300,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)
mesh = axes[2, 1].pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
axes[2, 1].set_ylim((0, 300))
axes[2, 1].axvline(time[int(np.fix(n_time_samples / 2))], color='black')

plt.tight_layout()
cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal',
                  shrink=.5, aspect=15, pad=0.1, label='Power')
cb.outline.set_linewidth(0)

####  With trial structure (time x trials)
time_halfbandwidth_product = 1

frequency_of_interest = [200, 50]
time_extent = (0, 0.600)
n_trials = 100
sampling_frequency = 1500
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
signal[:n_time_samples // 2, 1] = 0
signal = signal.sum(axis=1)[:, np.newaxis, np.newaxis] # (901, 1, 1)
noise = np.random.normal(0, 2, size=(n_time_samples, n_trials, 1)) # (901, 100, 1)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
axes[0, 0].plot(time, signal.squeeze())
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Signal', fontweight='bold')
axes[0, 0].set_ylim((-10, 10))

axes[0, 1].plot(time, signal[:, 0, 0] + noise[:, 0, 0])
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_title('Signal + Noise', fontweight='bold')
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
axes[1, 0].plot(c.frequencies, c.power().squeeze())

m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
axes[1, 1].plot(c.frequencies, c.power().squeeze())


m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               time_window_duration=0.060,
               time_window_step=0.060,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)
mesh = axes[2, 0].pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
axes[2, 0].set_ylim((0, 300))
axes[2, 0].axvline(time[int(np.fix(n_time_samples / 2))], color='black')

m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               time_window_duration=0.060,
               time_window_step=0.060,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)
mesh = axes[2, 1].pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
axes[2, 1].set_ylim((0, 300))
axes[2, 1].axvline(time[int(np.fix(n_time_samples / 2))], color='black')

plt.tight_layout()
cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal',
                  shrink=.5, aspect=15, pad=0.1, label='Power')
cb.outline.set_linewidth(0)
print('frequency resolution: {}'.format(m.frequency_resolution))

####   Decrease frequency resolution by decreasing time_halfbandwidth
time_halfbandwidth_product = 3

frequency_of_interest = [200, 50]
time_extent = (0, 0.600)
n_trials = 100
sampling_frequency = 1500
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
signal[:n_time_samples // 2, 1] = 0
signal = signal.sum(axis=1)[:, np.newaxis, np.newaxis]
noise = np.random.normal(0, 2, size=(n_time_samples, n_trials, 1))

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
axes[0, 0].plot(time, signal.squeeze())
axes[0, 0].set_xlim(time_extent)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Signal', fontweight='bold')
axes[0, 0].set_ylim((-10, 10))

axes[0, 1].plot(time, signal[:, 0, 0] + noise[:, 0, 0])
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_title('Signal + Noise', fontweight='bold')
axes[0, 1].set_xlim(time_extent)
axes[0, 1].set_ylim((-10, 10))

m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
axes[1, 0].plot(c.frequencies, c.power().squeeze())

m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
axes[1, 1].plot(c.frequencies, c.power().squeeze())


m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               time_window_duration=0.060,
               time_window_step=0.060,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)
mesh = axes[2, 0].pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
axes[2, 0].set_ylim((0, 300))
axes[2, 0].axvline(time[int(np.fix(n_time_samples / 2))], color='black')

m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               time_window_duration=0.060,
               time_window_step=0.060,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)
mesh = axes[2, 1].pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
axes[2, 1].set_ylim((0, 300))
axes[2, 1].axvline(time[int(np.fix(n_time_samples / 2))], color='black')

plt.tight_layout()
cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal',
                  shrink=.5, aspect=15, pad=0.1, label='Power')
cb.outline.set_linewidth(0)
print('frequency resolution: {}'.format(m.frequency_resolution))

####  Coherence: No trials, 200 Hz,  π/2 phase offset
time_halfbandwidth_product = 5
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 50)
n_signals = 2
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signal = np.zeros((n_time_samples, n_signals))
signal[:, 0] = np.sin(2 * np.pi * time * frequency_of_interest)
phase_offset = np.pi / 2
signal[:, 1] = np.sin((2 * np.pi * time * frequency_of_interest) + phase_offset)
noise = np.random.normal(0, 4, signal.shape)

plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.title('Signal', fontweight='bold')
plt.plot(time, signal[:, 0], label='Signal1')
plt.plot(time, signal[:, 1], label='Signal2')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))
plt.legend()

plt.subplot(2, 2, 2)
plt.title('Signal + Noise', fontweight='bold')
plt.plot(time, signal + noise)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim((0.95, 1.05))
plt.ylim((-10, 10))
plt.legend()

m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
plt.subplot(2, 2, 3)
plt.plot(c.frequencies, c.coherence_magnitude()[0, :, 0, 1])


m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
plt.subplot(2, 2, 4)
plt.plot(c.frequencies, c.coherence_magnitude()[0, :, 0, 1])

####  With trial structure (time x trials), 200 Hz, π/2 phase offset
time_halfbandwidth_product = 5
frequency_of_interest = 200
sampling_frequency = 1500
time_extent = (0, 0.600)
n_trials = 100
n_signals = 2
n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)[:, np.newaxis]
signal = np.zeros((n_time_samples, n_trials, n_signals))
signal[:, :, 0] = np.sin(2 * np.pi * time * frequency_of_interest)
phase_offset = np.pi / 2
signal[:, :, 1] = np.sin((2 * np.pi * time * frequency_of_interest) + phase_offset)
noise = np.random.normal(0, 4, signal.shape)

plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.title('Signal', fontweight='bold')
plt.plot(time, signal[:, 0, 0], label='Signal1')
plt.plot(time, signal[:, 0, 1], label='Signal2')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(time_extent)
plt.ylim((-2, 2))
plt.legend()

plt.subplot(2, 2, 2)
plt.title('Signal + Noise', fontweight='bold')
plt.plot(time, signal[:, 0, :] + noise[:, 0, :])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(time_extent)
plt.ylim((-10, 10))

m = Multitaper(signal,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
plt.subplot(2, 2, 3)
plt.plot(c.frequencies, c.coherence_magnitude()[0, :, 0, 1])


m = Multitaper(signal + noise,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
plt.subplot(2, 2, 4)
plt.plot(c.frequencies, c.coherence_magnitude()[0, :, 0, 1])
