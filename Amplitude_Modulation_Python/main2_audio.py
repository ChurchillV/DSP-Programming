import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import time

def amplitude_modulate_audio(audio_path, carrier_frequency, modulation_index):
    """
    Performs Amplitude Modulation on an audio signal.

    Args:
        audio_path (str): Path to the input WAV audio file.
        carrier_frequency (float): Frequency of the carrier wave in Hz.
        modulation_index (float): Modulation index (m), typically between 0 and 1.
                                 m = 0 indicates no modulation, m = 1 indicates full modulation.

    Returns:
        tuple: A tuple containing:
            - modulated_signal (numpy.ndarray): The AM modulated audio signal.
            - samplerate (int): The sample rate of the audio.
    """
    samplerate, audio_data = wavfile.read(audio_path)

    # Convert audio data to float if it's not already
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float64) / 32768.0  # Normalize to -1 to 1

    duration = len(audio_data) / samplerate
    time = np.linspace(0, duration, len(audio_data), endpoint=False)

    # Carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_frequency * time)

    # AM signal: s(t) = Ac * (1 + m * message_signal(t)) * cos(2 * pi * fc * t)
    # Assuming carrier amplitude Ac = 1 for simplicity here
    modulated_signal = (1 + modulation_index * audio_data) * carrier_wave

    return modulated_signal, samplerate

# Example usage:
if __name__ == "__main__":
    # Create a dummy audio file for demonstration if you don't have one
    # This creates a simple sine wave audio
    fs = 44100  # Sample rate
    duration_sec = 3  # seconds
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    dummy_audio = 0.8 * np.sin(2 * np.pi * 440 * t) # 440 Hz sine wave
    wavfile.write("dummy_audio.wav", fs, dummy_audio.astype(np.float32))

    input_audio_file = "dummy_audio.wav"
    carrier_freq = 10000  # Hz
    mod_index = 0.8

    am_signal, sr = amplitude_modulate_audio(input_audio_file, carrier_freq, mod_index)

    # Save the modulated audio
    wavfile.write("modulated_audio.wav", sr, am_signal.astype(np.float32))

    print(f"Modulated audio saved to 'modulated_audio.wav' with sample rate {sr} Hz.")

    # Plot a segment of the signals for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(time[:sr//100], dummy_audio[:sr//100], label='Original Audio (Message Signal)')
    plt.plot(time[:sr//100], am_signal[:sr//100], label='AM Modulated Signal')
    plt.title('Amplitude Modulation of Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()