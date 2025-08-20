import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 10000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector from 0 to 1 second

# Message signal (modulating signal)
fm = 10  # Message signal frequency (Hz)
Am = 1   # Message signal amplitude
message_signal = Am * np.cos(2 * np.pi * fm * t)

# Carrier signal
fc = 100  # Carrier signal frequency (Hz)
Ac = 1    # Carrier signal amplitude
carrier_signal = Ac * np.cos(2 * np.pi * fc * t)

# Amplitude Modulated signal
# s(t) = Ac * (1 + k_a * m(t)) * cos(2 * pi * fc * t)
# For simplicity, we can consider k_a * m(t) as the normalized message signal
# Here, we'll use a direct multiplication for demonstration of basic AM
# assuming the message signal is scaled appropriately to avoid overmodulation.
k_a = 0.8 # Modulation index (adjust for depth of modulation)
am_signal = Ac * (1 + k_a * message_signal) * carrier_signal

# Plotting the signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, message_signal)
plt.title('Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, carrier_signal)
plt.title('Carrier Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, am_signal)
plt.title('Amplitude Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()