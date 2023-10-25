import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

amplitude = 1.0
frequency = 1.0
duration = 1.0

t = np.linspace(0, 1, 1000)

carrier_signal = amplitude * np.sin(2 * np.pi * frequency * t)

modulating_signal = np.sign(np.sin(2 * np.pi * frequency * t))


def amplitude_modulation(carrier, modulating):
    return (1 + modulating) * carrier


def frequency_modulation(carrier, modulating, modulation_index):
    return carrier * np.cos(2 * np.pi * frequency * t + 2 * np.pi * modulation_index * modulating)


def phase_modulation(carrier, modulating, modulation_index):
    return carrier * np.cos(2 * np.pi * frequency * t + modulation_index * modulating)


amplitude_modulated_signal = amplitude_modulation(carrier_signal, modulating_signal)
frequency_modulated_signal = frequency_modulation(carrier_signal, modulating_signal, 1.0)
phase_modulated_signal = phase_modulation(carrier_signal, modulating_signal, 1.0)


def plot_spectrum(signal, title):
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(t), t[1] - t[0])
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label=title)
    plt.title(title)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, np.abs(spectrum))
    plt.title('Спектр ' + title)
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.tight_layout()
    plt.xlim(0, 10)


plot_spectrum(amplitude_modulated_signal, 'Амплитудная модуляция')
plot_spectrum(frequency_modulated_signal, 'Частотная модуляция')
plot_spectrum(phase_modulated_signal, 'Фазовая модуляция')

plt.show()

spectrum_amplitude_modulation = np.fft.fft(amplitude_modulated_signal)

cutoff_frequency = 50
spectrum_amplitude_modulation[:cutoff_frequency] = 0
spectrum_amplitude_modulation[-cutoff_frequency:] = 0

synthesized_signal = np.fft.ifft(spectrum_amplitude_modulation)

plt.figure()
plt.plot(t, np.real(synthesized_signal))
plt.title('Синтезированный сигнал (после обрезки спектра)')

plt.show()


def filter_signal(signal, cutoff_frequency):
    sampling_frequency = len(t) / duration
    nyquist = 0.5 * sampling_frequency
    low = cutoff_frequency / nyquist
    b, a = butter(1, low, btype='low')
    return filtfilt(b, a, signal)


cutoff_frequency_filter = 1
filtered_synthesized_signal = filter_signal(synthesized_signal, cutoff_frequency_filter)

plt.figure()
plt.plot(t, np.real(filtered_synthesized_signal), label='Фильтрованный сигнал')
plt.plot(t, modulating_signal, label='Модулирующий сигнал')
plt.title('Фильтрация синтезированного сигнала')
plt.legend()
plt.show()
