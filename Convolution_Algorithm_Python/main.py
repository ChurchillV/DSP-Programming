from utilities import voice_recorder_plotter_module as recorder
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import sounddevice as sd

def convolution(x, h):
    """
    Performs convolution and returns y[n] = x[n]*h[n]
    """
    output_size = len(x) + len(h) - 1
    y = [0 for _ in range(output_size)]

    for i in range(0, output_size):
        j = i
        while j >= 0:
            if(len(h) > (i - j) and len(x) > j):
                y[i] += x[j]*h[i-j]
            else:
                y[i] += 0
            j -= 1

    return y

def main():
    """
    Main function that runs convolution on an audio recording
    """
    print("Audio signal Convolution simulator")
    print("=================================")
    
    vr = recorder.VoiceRecorderPlotter(sample_rate=44100)

    # List available devices
    try:
        vr.list_audio_devices()
    except Exception as e:
        print(f"Could not list audio devices: {e}")
    
    print("\nOptions:")
    print("1. Record for fixed duration (5 seconds)")
    print("2. Manual start/stop recording")
    print("3. Test microphone")

    try:
        choice = input("\nEnter choice (1-3): ").strip()
        audio_data = []
        
        if choice == "1":
            duration = float(input("Enter recording duration in seconds (default 5): ") or "5")
            audio_data = vr.record_audio(duration=duration)

            
        elif choice == "2":
            audio_data = vr.record_audio_manual()            
            
        elif choice == "3":
            print("Testing microphone for 3 seconds...")
            test_data = vr.record_audio(3)
            max_level = np.max(np.abs(test_data))
            print(f"Maximum input level: {max_level:.3f}")
            if max_level < 0.001:
                print("Warning: Very low input level. Check microphone connection.")
            else:
                print("Microphone test successful!")
                
        else:
            print("Invalid choice!")

        
        if len(audio_data) != 0:
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            samplerate = 44100

            # Simple low pass FIR filter kernel
            h = np.ones(200) / 200

            # Convolution
            # y = np.array(convolution(audio_data, h), dtype=float)
            y = fftconvolve(audio_data, h, mode="full")

            # Normalize 
            y = y / np.max(np.abs(y))

            # Save output to file
            wavfile.write("output.wav", samplerate, (y * 32767).astype(np.int16))

            # Plot original vs filtered
            plt.figure(figsize=(12,6))
            plt.subplot(2,1,1)
            plt.title("Original Signal")
            plt.plot(audio_data[:2000])  # plot first 2000 samples for clarity
            plt.subplot(2,1,2)
            plt.title("Filtered Signal")
            plt.plot(y[:2000])
            plt.tight_layout()
            plt.show()

            # Play sound
            print("Playing processed audio...")
            sd.play(y, samplerate)
            sd.wait()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a working microphone connected.")

if __name__ == "__main__":
    main()