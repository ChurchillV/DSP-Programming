import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal
from scipy.fft import fft, fftfreq
import threading
import time

class VoiceRecorderPlotter:
    def __init__(self, sample_rate=44100, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = None
        self.is_recording = False
        
    def record_audio(self, duration=5):
        """
        Record audio from microphone for specified duration
        """
        print(f"Recording for {duration} seconds... Speak now!")
        print("3... 2... 1... GO!")
        
        # Record audio
        self.recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float64'
        )
        sd.wait()  # Wait until recording is finished
        
        print("Recording finished!")
        return self.recording.flatten()  # Convert to 1D array
    
    def record_audio_manual(self):
        """
        Record audio with manual start/stop control
        """
        print("Press Enter to start recording...")
        input()
        
        print("Recording... Press Enter to stop.")
        
        # Start recording in a separate thread
        recorded_data = []
        self.is_recording = True
        
        def recording_callback(indata, frames, time, status):
            if self.is_recording:
                recorded_data.extend(indata.flatten())
        
        # Start the audio stream
        with sd.InputStream(callback=recording_callback, 
                          samplerate=self.sample_rate, 
                          channels=self.channels):
            input()  # Wait for user to press Enter
        
        self.is_recording = False
        self.recording = np.array(recorded_data)
        
        print("Recording stopped!")
        return self.recording
    
    def plot_waveform(self, audio_data):
        """
        Plot the time-domain waveform
        """
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, audio_data, 'b-', linewidth=0.5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('Voice Recording - Time Domain')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_frequency_spectrum(self, audio_data):
        """
        Plot the frequency spectrum using FFT
        """
        # Compute FFT
        fft_data = fft(audio_data)
        frequencies = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Only plot positive frequencies
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = np.abs(fft_data[positive_freq_idx])
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.semilogy(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Voice Recording - Frequency Spectrum (Linear Scale)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 4000)  # Focus on speech frequency range
        
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:len(frequencies)//2], 20*np.log10(magnitude[:len(magnitude)//2] + 1e-10))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Voice Recording - Frequency Spectrum (dB Scale)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 4000)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_spectrogram(self, audio_data):
        """
        Plot spectrogram showing frequency content over time
        """
        # Compute spectrogram
        frequencies, times, Sxx = scipy.signal.spectrogram(
            audio_data, 
            self.sample_rate, 
            nperseg=1024,
            noverlap=512
        )
        
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, frequencies, 10*np.log10(Sxx + 1e-10), 
                      shading='gouraud', cmap='viridis')
        plt.colorbar(label='Power (dB)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (seconds)')
        plt.title('Voice Recording - Spectrogram')
        plt.ylim(0, 4000)  # Focus on speech frequency range
        plt.tight_layout()
        return plt.gcf()
    
    def plot_envelope(self, audio_data):
        """
        Plot the amplitude envelope of the recording
        """
        # Calculate envelope using Hilbert transform
        analytic_signal = scipy.signal.hilbert(audio_data)
        envelope = np.abs(analytic_signal)
        
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, audio_data, 'b-', alpha=0.5, linewidth=0.5, label='Waveform')
        plt.plot(time_axis, envelope, 'r-', linewidth=2, label='Envelope')
        plt.plot(time_axis, -envelope, 'r-', linewidth=2)
        plt.fill_between(time_axis, envelope, -envelope, alpha=0.2, color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('Voice Recording - Amplitude Envelope')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_voice_characteristics(self, audio_data):
        """
        Analyze and plot voice characteristics
        """
        # Calculate RMS energy over time
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        rms_energy = []
        zero_crossing_rate = []
        time_frames = []
        
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            
            # RMS Energy
            rms = np.sqrt(np.mean(frame**2))
            rms_energy.append(rms)
            
            # Zero Crossing Rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            zero_crossing_rate.append(zcr)
            
            time_frames.append(i / self.sample_rate)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Original waveform
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
        axes[0].plot(time_axis, audio_data, 'b-', linewidth=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Original Waveform')
        axes[0].grid(True, alpha=0.3)
        
        # RMS Energy
        axes[1].plot(time_frames, rms_energy, 'r-', linewidth=2)
        axes[1].set_ylabel('RMS Energy')
        axes[1].set_title('RMS Energy (Voice Activity)')
        axes[1].grid(True, alpha=0.3)
        
        # Zero Crossing Rate
        axes[2].plot(time_frames, zero_crossing_rate, 'g-', linewidth=2)
        axes[2].set_ylabel('Zero Crossing Rate')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_title('Zero Crossing Rate (Voiced/Unvoiced)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def comprehensive_analysis(self, duration=5, manual_control=False):
        """
        Record audio and create comprehensive analysis plots
        """
        # Record audio
        if manual_control:
            audio_data = self.record_audio_manual()
        else:
            audio_data = self.record_audio(duration)
        
        if len(audio_data) == 0:
            print("No audio data recorded!")
            return
        
        # Normalize audio data
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        print(f"Recorded {len(audio_data)/self.sample_rate:.2f} seconds of audio")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Max amplitude: {np.max(np.abs(audio_data)):.3f}")
        print(f"RMS amplitude: {np.sqrt(np.mean(audio_data**2)):.3f}")
        
        # Create all plots
        print("Generating plots...")
        
        self.plot_waveform(audio_data)
        plt.show()
        
        self.plot_frequency_spectrum(audio_data)
        plt.show()
        
        self.plot_spectrogram(audio_data)
        plt.show()
        
        self.plot_envelope(audio_data)
        plt.show()
        
        self.analyze_voice_characteristics(audio_data)
        plt.show()
        
        return audio_data

def list_audio_devices():
    """
    List available audio input devices
    """
    print("Available audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (Max inputs: {device['max_input_channels']})")

def main():
    """
    Main function to run the voice recorder and plotter
    """
    print("Voice Recording and Analysis Tool")
    print("=================================")
    
    # List available devices
    try:
        list_audio_devices()
    except Exception as e:
        print(f"Could not list audio devices: {e}")
    
    print("\nOptions:")
    print("1. Record for fixed duration (5 seconds)")
    print("2. Manual start/stop recording")
    print("3. Test microphone")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        recorder = VoiceRecorderPlotter(sample_rate=44100)
        
        if choice == "1":
            duration = float(input("Enter recording duration in seconds (default 5): ") or "5")
            recorder.comprehensive_analysis(duration=duration, manual_control=False)
            
        elif choice == "2":
            recorder.comprehensive_analysis(manual_control=True)
            
        elif choice == "3":
            print("Testing microphone for 3 seconds...")
            test_data = recorder.record_audio(3)
            max_level = np.max(np.abs(test_data))
            print(f"Maximum input level: {max_level:.3f}")
            if max_level < 0.001:
                print("Warning: Very low input level. Check microphone connection.")
            else:
                print("Microphone test successful!")
                
        else:
            print("Invalid choice!")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a working microphone connected.")

if __name__ == "__main__":
    main()