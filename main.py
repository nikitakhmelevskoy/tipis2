import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert


def square_wave(frequency, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.heaviside(np.sin(2 * np.pi * frequency * t), 0)
    return t, signal


def amplitude_modulation(carrier_freq, modulation_freq, duration, sampling_rate):
    t, message_signal = square_wave(modulation_freq, duration, sampling_rate)
    carrier_signal = np.sin(2 * np.pi * carrier_freq * t)
    modulated_signal = (1 + 0.5 * message_signal) * carrier_signal
    return t, modulated_signal


def frequency_modulation(carrier_freq, modulation_freq, duration, sampling_rate):
    t, message_signal = square_wave(modulation_freq, duration, sampling_rate)
    deviation = 200
    carrier_signal = np.sin(2 * np.pi * carrier_freq * t + deviation * np.cumsum(message_signal) / sampling_rate)
    return t, carrier_signal


def phase_modulation(carrier_freq, modulation_freq, duration, sampling_rate):
    t, message_signal = square_wave(modulation_freq, duration, sampling_rate)
    phase_deviation = 0.5
    deviation = phase_deviation * 2 * np.pi * message_signal
    carrier_signal = np.sin(2 * np.pi * carrier_freq * t + deviation)
    return t, carrier_signal


def synthesize_from_spectrum(truncated_spectrum):
    return np.real(ifft(truncated_spectrum))


duration = 1
sampling_rate = 1000
carrier_freq = 50
modulation_freq = 5

t, am_signal = amplitude_modulation(carrier_freq, modulation_freq, duration, sampling_rate)
t, fm_signal = frequency_modulation(carrier_freq, modulation_freq, duration, sampling_rate)
t, pm_signal = phase_modulation(carrier_freq, modulation_freq, duration, sampling_rate)

spectrum = fft(am_signal)
low_cutoff = 40
high_cutoff = 60

truncated_spectrum = spectrum.copy()
truncated_spectrum[:low_cutoff] = 0
truncated_spectrum[high_cutoff:] = 0
freq = np.fft.fftfreq(len(spectrum), d=1 / sampling_rate)

synthesized_signal = synthesize_from_spectrum(truncated_spectrum)

analytic_signal = hilbert(synthesized_signal)

restored_signal = np.abs(analytic_signal)

threshold = 0.5

filtered_restored = np.where(restored_signal > threshold, 1, 0)

plt.figure(figsize=(12, 8))

plt.subplot(4, 2, 1)
plt.plot(t, am_signal)
plt.title('Амплитудная модуляция')

plt.subplot(4, 2, 2)
plt.magnitude_spectrum(am_signal, Fs=sampling_rate)
plt.title('Спектр амплитудной модуляции')
plt.xlim(0, 150)

plt.subplot(4, 2, 3)
plt.plot(t, fm_signal)
plt.title('Частотная модуляция')

plt.subplot(4, 2, 4)
plt.magnitude_spectrum(fm_signal, Fs=sampling_rate)
plt.title('Спектр частотной модуляции')
plt.xlim(0, 150)

plt.subplot(4, 2, 5)
plt.plot(t, pm_signal)
plt.title('Фазовая модуляция')

plt.subplot(4, 2, 6)
plt.magnitude_spectrum(pm_signal, Fs=sampling_rate)
plt.title('Спектр фазовой модуляции')
plt.xlim(0, 150)

plt.subplot(4, 2, 7)
plt.plot(t, synthesized_signal, label='Синтезированный сигнал из обрезанного спектра')
plt.title('Синтезированный сигнал')

plt.subplot(4, 2, 8)
plt.plot(freq[:len(freq) // 2], abs(truncated_spectrum[:len(freq) // 2]),
         label='Обрезанный спектр амплитудной модуляции')
plt.title('Обрезанный спектр амплитудной модуляции')
plt.xlabel('Frequency')
plt.ylabel('Magnitude (energy)')
plt.xlim(0, 150)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.plot(t, filtered_restored, label='Отфильтрованный сигнал')
plt.title('Отфильтрованный')
plt.xlabel('Время')
plt.ylabel('Амплитуда')

plt.tight_layout()
plt.show()