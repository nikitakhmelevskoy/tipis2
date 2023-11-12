import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
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
    carrier_signal = np.sin(2 * np.pi * carrier_freq * t + phase_deviation * np.cumsum(message_signal) / sampling_rate)
    return t, carrier_signal


def synthesize_from_spectrum(truncated_spectrum):
    return np.real(ifft(truncated_spectrum))


def synthesize_matching_signal(modulation_freq, duration, sampling_rate):
    t, message_signal = square_wave(modulation_freq, duration, sampling_rate)
    synthesized_signal = np.tile(message_signal, int(sampling_rate * duration / len(message_signal)))
    return t, synthesized_signal[:int(sampling_rate * duration)]


def filter_and_match_shape(modulated_signal, modulating_signal):
    filtered_signal = np.convolve(modulated_signal, modulating_signal, mode='same') / np.sum(modulating_signal)
    return filtered_signal


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

spectrum[:low_cutoff] = 0
spectrum[high_cutoff:] = 0

synthesized_signal = synthesize_from_spectrum(spectrum)

filtered_synthesized_signal = filter_and_match_shape(square_wave(modulation_freq, duration, sampling_rate)[1], synthesized_signal)

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
plt.legend()

plt.subplot(4, 2, 8)
plt.plot(t, square_wave(modulation_freq, duration, sampling_rate)[1], label='Модулирующий сигнал (меандр)', linewidth=2)
plt.title('Модулирующий сигнал (однополярный меандр)')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда')
plt.legend()
plt2 = plt.twinx()
plt2.plot(t, filtered_synthesized_signal,'y')

plt.tight_layout()
plt.show()